from __future__ import annotations

import copy
import json
import random
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import requests

from .models import Environment, Waveform, WaveformConstraints
from .rag import LibraryEntry, RuleBasedDesigner


class LLMAPIError(RuntimeError):
    """Raised when the API request fails after all retry attempts."""

    def __init__(self, message: str, *, status_code: int | None = None, retryable: bool = False):
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable


@dataclass
class LLMUsage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_sec: float = 0.0
    parse_success_count: int = 0
    parsed_candidate_count: int = 0
    fallback_count: int = 0
    api_error_count: int = 0          # final failures, not transient retries that eventually succeed
    retry_count: int = 0              # transient retry attempts
    api_attempt_count: int = 0        # total HTTP attempts
    parse_error_count: int = 0
    truncated_or_max_tokens_count: int = 0
    last_status_code: int = 0
    last_error: str = ""

    def to_dict(self) -> dict:
        return {
            "llm_calls": self.calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "llm_latency_sec": self.latency_sec,
            "parse_success_count": self.parse_success_count,
            "parsed_candidate_count": self.parsed_candidate_count,
            "fallback_count": self.fallback_count,
            "api_error_count": self.api_error_count,
            "retry_count": self.retry_count,
            "api_attempt_count": self.api_attempt_count,
            "parse_error_count": self.parse_error_count,
            "truncated_or_max_tokens_count": self.truncated_or_max_tokens_count,
            "last_status_code": self.last_status_code,
            "last_error": self.last_error,
        }

    def copy(self) -> "LLMUsage":
        return copy.deepcopy(self)


class LLMWaveformClient:
    """Direct HTTP client for OpenAI-compatible chat completion APIs.

    v3.1 adds robust retry/backoff support for 502/503/504/429/timeouts and
    other transient failures. Full LLM methods still do not silently become
    no-API methods unless allow_rule_fallback=True.
    """

    def __init__(
        self,
        model: str,
        api_url: str,
        api_key: str,
        use_api: bool = True,
        require_api: bool = True,
        allow_rule_fallback: bool = False,
        temperature: float = 0.25,
        max_tokens: int = 2500,
        timeout_sec: int = 180,
        use_response_format_json: bool = True,
        constraints: WaveformConstraints | None = None,
        seed: int = 0,
        trace_dir: str | Path | None = None,
        trace_tag: str = "llm_call",
        max_retries: int = 8,
        retry_initial_delay: float = 2.0,
        retry_max_delay: float = 120.0,
        retry_exp_base: float = 2.0,
        retry_jitter: float = 0.25,
        retry_status_codes: list[int] | tuple[int, ...] | None = None,
        parse_retry_max: int = 2,
    ):
        self.model = model
        self.api_url = api_url
        self.api_key = api_key
        self.use_api = bool(use_api)
        self.require_api = bool(require_api)
        self.allow_rule_fallback = bool(allow_rule_fallback)
        self.temperature = temperature
        self.max_tokens = int(max_tokens)
        self.timeout_sec = int(timeout_sec)
        self.use_response_format_json = bool(use_response_format_json)
        self.constraints = constraints or WaveformConstraints()
        self.rule = RuleBasedDesigner(self.constraints, seed=seed)
        self.usage = LLMUsage()
        self.trace_dir = Path(trace_dir) if trace_dir else None
        self.trace_tag = trace_tag
        self._call_index = 0
        self.rng = random.Random(seed)

        self.max_retries = int(max_retries)
        self.retry_initial_delay = float(retry_initial_delay)
        self.retry_max_delay = float(retry_max_delay)
        self.retry_exp_base = float(retry_exp_base)
        self.retry_jitter = float(retry_jitter)
        self.retry_status_codes = set(retry_status_codes or [408, 409, 425, 429, 500, 502, 503, 504])
        self.parse_retry_max = int(parse_retry_max)

    def is_configured(self) -> bool:
        return bool(self.api_url and self.model and self.api_key and "PASTE_YOUR_API_KEY" not in self.api_key)

    def propose(self, env: Environment, retrieved: List[LibraryEntry], n: int = 6) -> List[Waveform]:
        if not self.use_api:
            if self.require_api:
                raise RuntimeError("LLM API is required for this method but use_api=False.")
            self.usage.fallback_count += 1
            return self.rule.propose(env, n=n)

        if not self.is_configured():
            msg = "LLM API is not configured. Edit API_URL, API_KEY and MODEL_ID in radar_llm_robust/config.py."
            if self.require_api and not self.allow_rule_fallback:
                self.usage.api_error_count += 1
                self.usage.last_error = msg
                raise RuntimeError(msg)
            self.usage.fallback_count += 1
            return self.rule.propose(env, n=n)

        base_prompt = self._build_prompt(env, retrieved, n)
        last_exc: Exception | None = None

        for parse_round in range(self.parse_retry_max + 1):
            prompt = base_prompt
            if parse_round > 0:
                prompt = (
                    base_prompt
                    + "\n\nPrevious response could not be parsed. Return ONLY one JSON object with a top-level "
                      "'candidates' list. No markdown fences and no explanatory prose."
                )
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": self._system_prompt()},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }
            if self.use_response_format_json:
                payload["response_format"] = {"type": "json_object"}

            try:
                data, status_code, raw_text, attempts = self._post_with_retry(payload)
                self.usage.calls += 1
                usage = data.get("usage", {}) if isinstance(data, dict) else {}
                self.usage.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
                self.usage.completion_tokens += int(usage.get("completion_tokens", 0) or 0)

                choice = data["choices"][0]
                finish_reason = str(choice.get("finish_reason", "") or "")
                if finish_reason in {"length", "max_tokens"}:
                    self.usage.truncated_or_max_tokens_count += 1
                text = choice["message"].get("content", "") or ""
                waves = self._parse(text)
                self._write_trace(payload, status_code, raw_text, text, None, len(waves), finish_reason, attempts)
                if waves:
                    self.usage.parse_success_count += 1
                    self.usage.parsed_candidate_count += len(waves)
                    return [self.constraints.repair(w, env) for w in waves[:n]]

                self.usage.parse_error_count += 1
                last_exc = RuntimeError("LLM response was received but no valid waveform candidates were parsed.")
            except Exception as exc:
                last_exc = exc
                # _post_with_retry has already counted final API failure.
                self._write_trace(payload, getattr(exc, "status_code", None), None, "", repr(exc), 0, "", [])

            # Parse failures can be retried with a stricter prompt. Transport/API failures have already
            # used all retry attempts inside _post_with_retry, so do not repeat the whole retry budget here.
            if isinstance(last_exc, LLMAPIError):
                break

        if self.require_api and not self.allow_rule_fallback:
            raise last_exc if last_exc is not None else RuntimeError("LLM API failed without a detailed exception.")
        self.usage.fallback_count += 1
        return self.rule.propose(env, n=n)

    def _post_with_retry(self, payload: dict) -> Tuple[dict, int, str, list[dict]]:
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"}
        attempts: list[dict] = []
        payload_current = dict(payload)
        response_format_removed = False
        max_attempts = self.max_retries + 1
        start = time.perf_counter()

        for attempt in range(1, max_attempts + 1):
            attempt_t0 = time.perf_counter()
            self.usage.api_attempt_count += 1
            status_code: int | None = None
            raw_text = ""
            error_text = ""
            retryable = False
            retry_after_delay: float | None = None

            try:
                response = requests.post(self.api_url, headers=headers, json=payload_current, timeout=self.timeout_sec)
                status_code = int(response.status_code)
                raw_text = response.text or ""
                self.usage.last_status_code = status_code

                # Some OpenAI-compatible endpoints reject response_format. Remove it once and retry immediately.
                if (
                    status_code == 400
                    and "response_format" in payload_current
                    and (("response_format" in raw_text.lower()) or ("json_object" in raw_text.lower()) or ("json" in raw_text.lower()))
                    and not response_format_removed
                ):
                    response_format_removed = True
                    payload_current = dict(payload_current)
                    payload_current.pop("response_format", None)
                    error_text = "Endpoint rejected response_format; retrying without response_format."
                    retryable = True
                elif 200 <= status_code < 300:
                    try:
                        data = response.json()
                    except Exception as exc:
                        error_text = f"Invalid JSON response: {exc}"
                        retryable = True
                    else:
                        self.usage.latency_sec += time.perf_counter() - start
                        attempts.append({
                            "attempt": attempt,
                            "status_code": status_code,
                            "retryable": False,
                            "error": "",
                            "duration_sec": time.perf_counter() - attempt_t0,
                        })
                        return data, status_code, raw_text, attempts
                elif status_code in self.retry_status_codes:
                    retryable = True
                    error_text = f"HTTP {status_code}: {raw_text[:500]}"
                    retry_after_delay = self._retry_after_seconds(response.headers.get("Retry-After"))
                else:
                    error_text = f"Non-retryable HTTP {status_code}: {raw_text[:500]}"
                    retryable = False

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                retryable = True
            except requests.exceptions.RequestException as exc:
                error_text = f"{type(exc).__name__}: {exc}"
                retryable = True

            attempts.append({
                "attempt": attempt,
                "status_code": status_code,
                "retryable": retryable,
                "error": error_text,
                "duration_sec": time.perf_counter() - attempt_t0,
            })
            self.usage.last_error = error_text

            if not retryable:
                self.usage.latency_sec += time.perf_counter() - start
                self.usage.api_error_count += 1
                raise LLMAPIError(error_text, status_code=status_code, retryable=False)

            if attempt >= max_attempts:
                self.usage.latency_sec += time.perf_counter() - start
                self.usage.api_error_count += 1
                raise LLMAPIError(
                    f"API failed after {max_attempts} attempts. Last error: {error_text}",
                    status_code=status_code,
                    retryable=True,
                )

            self.usage.retry_count += 1
            delay = retry_after_delay if retry_after_delay is not None else self._backoff_delay(attempt)
            print(f"      transient API error; retry {attempt}/{max_attempts-1} after {delay:.1f}s: {error_text[:160]}")
            time.sleep(delay)

        # Unreachable, but helps static checkers.
        self.usage.api_error_count += 1
        raise LLMAPIError("API retry loop exited unexpectedly.", retryable=True)

    def _retry_after_seconds(self, value: str | None) -> float | None:
        if not value:
            return None
        try:
            seconds = float(value)
            return max(0.0, min(seconds, self.retry_max_delay))
        except Exception:
            return None

    def _backoff_delay(self, attempt: int) -> float:
        base_delay = self.retry_initial_delay * (self.retry_exp_base ** max(0, attempt - 1))
        jitter = base_delay * self.retry_jitter * self.rng.random()
        return min(self.retry_max_delay, base_delay + jitter)

    def _system_prompt(self) -> str:
        return (
            "You are a cognitive radar waveform design assistant. Generate feasible X-band pulsed radar waveform candidates. "
            "Do not claim optimality. Output JSON only. Use retrieved verified cases as priors and prefer robust designs "
            "under clutter, jamming, low SNR, and uncertainty."
        )

    def _build_prompt(self, env: Environment, retrieved: List[LibraryEntry], n: int) -> str:
        examples = []
        for e in retrieved[:6]:
            examples.append({
                "environment": e.env.to_dict(),
                "waveform": e.waveform.to_dict(),
                "score": round(e.score, 4),
                "rationale": e.rationale[:180],
            })
        schema = {
            "candidates": [
                {
                    "carrier_freq_ghz": 10.0,
                    "bandwidth_mhz": 220.0,
                    "prf_khz": 12.0,
                    "pulse_width_us": 8.0,
                    "modulation": "LFM",
                    "n_pulses": 64,
                    "rationale": "brief physical rationale"
                }
            ]
        }
        return (
            "Current radar environment:\n" + json.dumps(env.to_dict(), indent=2) + "\n\n"
            "Retrieved verified designs:\n" + json.dumps(examples, indent=2) + "\n\n"
            f"Return exactly {n} diverse waveform candidates. Constraints: "
            "carrier_freq_ghz in [8,12], bandwidth_mhz in [10,500], prf_khz in [1,100], "
            "pulse_width_us in [0.1,50], n_pulses in [8,128], modulation in [LFM,BPSK,Barker,Costas], "
            "and duty cycle prf_khz*pulse_width_us*1e-3 <= 0.20. "
            "Prefer robust worst-case detection and low violation rate, not only nominal score.\n"
            "Output strict JSON matching this schema:\n" + json.dumps(schema, indent=2)
        )

    def _parse(self, text: str) -> List[Waveform]:
        raw = (text or "").strip()
        objs = []
        try:
            objs.append(json.loads(raw))
        except Exception:
            pass
        for m in re.finditer(r"```(?:json)?\s*(\{.*?\})\s*```", raw, flags=re.DOTALL):
            try:
                objs.append(json.loads(m.group(1)))
            except Exception:
                pass
        if not objs:
            match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
            if match:
                try:
                    objs.append(json.loads(match.group(0)))
                except Exception:
                    pass

        waves: List[Waveform] = []
        for obj in objs:
            candidates = obj.get("candidates", []) if isinstance(obj, dict) else obj
            if isinstance(candidates, dict):
                candidates = [candidates]
            for c in candidates:
                if not isinstance(c, dict):
                    continue
                try:
                    waves.append(Waveform(
                        carrier_freq_ghz=float(c.get("carrier_freq_ghz", c.get("carrier_freq_GHz", 10.0))),
                        bandwidth_mhz=float(c.get("bandwidth_mhz", c.get("bandwidth_MHz", 100.0))),
                        prf_khz=float(c.get("prf_khz", c.get("prf_kHz", 10.0))),
                        pulse_width_us=float(c.get("pulse_width_us", 5.0)),
                        modulation=str(c.get("modulation", "LFM")),
                        n_pulses=int(c.get("n_pulses", 32)),
                    ))
                except Exception:
                    continue
        return waves

    def _write_trace(
        self,
        payload: dict,
        status_code: Optional[int],
        raw_text: Optional[str],
        content: str,
        error: Optional[str],
        parsed_count: int,
        finish_reason: str,
        attempts: list[dict] | None = None,
    ) -> None:
        if self.trace_dir is None:
            return
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self._call_index += 1
        safe_payload = dict(payload)
        trace = {
            "tag": self.trace_tag,
            "call_index": self._call_index,
            "api_url": self.api_url,
            "model": self.model,
            "status_code": status_code,
            "finish_reason": finish_reason,
            "attempts": attempts or [],
            "payload": safe_payload,
            "raw_response_text": raw_text,
            "message_content": content,
            "error": error,
            "parsed_candidate_count": parsed_count,
            "usage": self.usage.to_dict(),
        }
        path = self.trace_dir / f"{self.trace_tag}_{self._call_index:03d}.json"
        path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
