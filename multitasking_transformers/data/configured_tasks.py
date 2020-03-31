TASKS = {
    'ner': { }
    ,
    'sts': { }
    ,
    'nli': { }

}
clinical_ner = ['i2b2_2010', 'i2b2_2012', 'i2b2_2014', 'n2c2_2018', 'quaero_2014']
clinical_sts = ['n2c2_2019']
clinical_nli = ['medrqe_2016', 'mednli_2018']

misc_ner = ['tac_2018', 'end_2017']


biomedical_ner = ["AnatEM", "BC5CDR-disease", "BioNLP11ID-chem", "BioNLP13CG-cc", "BioNLP13CG",
                "BioNLP13PC-chem", "CRAFT-cell", "CRAFT-species", "NCBI-disease", "BC2GM", "BC5CDR",
                "BioNLP11ID-ggp", "BioNLP13CG-cell", "BioNLP13CG-species", "BioNLP13PC-ggp",
                "CRAFT-chem", "Ex-PTM", "BC4CHEMD", "BioNLP09", "BioNLP11ID", "BioNLP13CG-chem",
                "BioNLP13GE", "BioNLP13PC", "CRAFT-ggp", "JNLPBA", "BC5CDR-chem", "BioNLP11EPI",
                "BioNLP11ID-species", "BioNLP13CG-ggp", "BioNLP13PC-cc", "CRAFT-cc", "CRAFT", "linnaeus"]



def get_misc_configured_tasks(preprocessed_directory):
    for dataset in misc_ner:
        if dataset not in TASKS['ner']:
            TASKS['ner'][dataset] = {}
        TASKS['ner'][dataset].update(
            {
            'head': 'subword_classification',
            'batch_size': 25,
            'train': f"{preprocessed_directory}/{dataset}/ner/train",
            'test': f"{preprocessed_directory}/{dataset}/ner/test"
            }
        )

    return TASKS



def get_biomedical_configured_tasks(preprocessed_directory):
    # Update with biomedical ner tasks
    for dataset in biomedical_ner:
        if dataset not in TASKS['ner']:
            TASKS['ner'][dataset] = {}
        TASKS['ner'][dataset].update(
            {
                'head': 'subword_classification',
                'batch_size': 20,
                'train': f"{preprocessed_directory}/biomedical/ner/{dataset}/train",
                'test': f"{preprocessed_directory}/biomedical/ner/{dataset}/test",
                'evaluate_biluo': False
            }
        )
    return TASKS

def get_clinical_configured_tasks(preprocessed_directory):
    for dataset in clinical_ner:
        if dataset not in TASKS['ner']:
            TASKS['ner'][dataset] = {}
        TASKS['ner'][dataset].update(
            {
            'head': 'subword_classification',
            'batch_size': 25,
            'train': f"{preprocessed_directory}/{dataset}/ner/train",
            'test': f"{preprocessed_directory}/{dataset}/ner/test"
            }
        )

    for dataset in clinical_sts:
        if dataset not in TASKS['sts']:
            TASKS['sts'][dataset] = {}
        TASKS['sts'][dataset].update(
            {
            'head': 'cls_regression',
            'batch_size': 40,
            'train': f"{preprocessed_directory}/{dataset}/similarity/train",
            'test': f"{preprocessed_directory}/{dataset}/similarity/test"
            }
        )
    for dataset in clinical_nli:
        if dataset not in TASKS['nli']:
            TASKS['nli'][dataset] = {}
        TASKS['nli'][dataset].update(
            {
            'head': 'cls_classification',
            'batch_size': 40,
            'train': f"{preprocessed_directory}/{dataset}/nli/train",
            'test': f"{preprocessed_directory}/{dataset}/nli/test"
            }
        )

    return TASKS


huner_datasets = {
    'cellline': ['cll', 'gellus', 'jnlpba'],  #'cellfinder'],
    'chemical': ['cdr', 'cemp', 'chebi', 'chemdner', 'scai_chemicals'], #'biosemantics'],
    'disease': ['biosemantics', 'cdr', 'miRNA', 'ncbi', 'scai_diseases'],  # TODO add arizona
    'gene': ['bc2gm', 'bioinfer', 'cellfinder', 'deca', 'fsu', 'gpro', 'iepa', 'jnlpba', 'miRNA',
             'osiris', 'variome'],
    'species': ['cellfinder', 'linneaus', 'miRNA', 's800', 'variome']
}

def get_huner_tasks(preprocessed_directory):
    for task in huner_datasets:
        for dataset in huner_datasets[task]:
            if dataset in ['cellfinder', 'biosemantics']:  # TODO figure out the right batch size and delete this line -cellfinder. biosemantics is large. ignored for preliminary results.
                continue
            if dataset not in TASKS['ner']:
                TASKS['ner'][dataset] = {}
            TASKS['ner'][dataset].update(
                {
                'head': 'subword_classification',
                'batch_size': 20,
                'train': f"{preprocessed_directory}/biomedical/huner/{dataset}/train",
                'test': f"{preprocessed_directory}/biomedical/huner/{dataset}/test"
                }
            )
    return TASKS


def get_all_configured_tasks(preprocessed_directory):
    get_biomedical_configured_tasks(preprocessed_directory)
    get_clinical_configured_tasks(preprocessed_directory)
    get_misc_configured_tasks(preprocessed_directory)

    return TASKS
