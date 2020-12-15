"""Experiment running.
"""
import wandb
import runconfigs
# from patlib import Path


def prepare_dataset():

    taxo_count_limit = runconfigs.TAXO_COUNT_LIMIT
    sample_length_limit = runconfigs.SAMPLE_LENGTH_LIMIT
    taxonomy_file_path = runconfigs.TAXONOMY_FILE_PATH

    megan_labeled_files_info_path = runconfigs.MEGAN_LABELED_FILES_INFO_PATH

    csv4megan_excell_clenaed = runconfigs.CSV4MEGAN_EXCELL_CLEANED

    ignore_files = runconfigs.IGNORE_FILES

    excerpt_length = runconfigs.EXCERPT_LENGTH
    excell_names2code = runconfigs.EXCELL_NAMES2CODE

    audio_dataset, samples_2_delete, deleted_files = prepare_dataset.run(
        megan_labeled_files_info_path,
        taxonomy_file_path,
        csv4megan_excell_clenaed,
        ignore_files,
        excerpt_length,
        sample_length_limit,
        taxo_count_limit,
        excell_names2code=excell_names2code)
    
    return audio_dataset, samples_2_delete, deleted_files
    
def setup(config, wandb_project_name):

    wandb.init(config=config, project=wandb_project_name)
    config = wandb.config
    # wandb.config.update(args) # adds all of the arguments as config variables

    params = {
        "batch_size": config.batch_size,
        "shuffle": True,
        "num_workers": 0
    }

    audio_dataset,_,_ =prepare_dataset()




def main():
    wandb_project_name = runconfigs.PROJECT_NAME
    default_config = runconfigs.default_config
    setup(default_config, wandb_project_name)
