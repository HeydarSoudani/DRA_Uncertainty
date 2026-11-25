"""What every job of the package (download, indexing, training_datasets) and
the rest of the repo use:

    layout           canonical paths, data root, split-id + corpus-name rules
    dataset_loaders  read queries / qrels / answers / nuggets from that layout
    doc_lookup       fetch corpus documents by id (byte-offset index)

Nothing here knows a job: the jobs import from here, never the reverse.
"""
