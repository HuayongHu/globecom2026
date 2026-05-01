from radar_llm_robust import config
from radar_llm_robust.experiments import run_paper_experiment
from radar_llm_robust.plot_results import make_all_figures
from radar_llm_robust.semantic_stress import run_semantic_stress
from radar_llm_robust.paper_reporting_v45 import generate_paper_outputs


def main():
    print('=' * 78)
    print('RAG-CRA v4.5 full pipeline')
    print('Runs main experiments, optional semantic stress, and v4.5 paper reporting.')
    print('=' * 78)
    run_paper_experiment(config)
    make_all_figures(config.OUTPUT_DIR, save_pdf=bool(config.SAVE_PDF_FIGURES))
    if bool(getattr(config, 'RUN_SEMANTIC_STRESS', False)):
        run_semantic_stress(config)
    generate_paper_outputs(config.OUTPUT_DIR, getattr(config, 'SEMANTIC_OUTPUT_DIR', 'outputs/semantic_stress'))
    print('Completed. Paper-oriented tables are under:', config.OUTPUT_DIR + '/paper_v45')


if __name__ == '__main__':
    main()
