import pandas as pd 

def select_samples_via_metadata(my_selection, metadata):
    '''
    FUNCTION: Takes a metadata dataframe with columns as categories
                and rows as samples and outputs the sample names
                that meet your defined criteria
    INPUT: my_selection: a dictionary of key-value pairs from metadata
            'metdata: a pandas dataframe containing all metadata
                Ex: {'Tissue': 'Kidney', 'Prognosis': 'Bad'}
    OUTPUT: A list of the samples that meet the selected criteria
                Ex: ['patient 1', patient 12', 'patient 3'] all
                had bad kidneys
    '''
    selected_metadata = metadata # Not sure if this assignment is necessary
    for key, val in my_selection.iteritems():
        selected_metadata = selected_metadata[ (selected_metadata[key]
            == val)]

    selected_samples = selected_metadata.index
    return list(selected_samples)

def get_phylo_from_RDP(rdp_path, otu, cutoff):
    '''
    FUNTION - Grabs the most specific phylogenetic info that matches your cutoff
    INPUT - rdp_path: the path to your rdp file, 
        otu: a string 
        cutoff: The probability cutoff, between 0 and 1
    OUTPUT - a string stating the phylogenetic assignment
    '''
    # import rdp, with denovo otus as rows
    column_names = ['empty_1', 'empty_2', 'empty_3', 'empty_4',
    'Kingdom', 'Kingdom_2', 'Kingdom_score',
    'Phylum', 'Phylum_2', 'Phylum_score',
    'Class', 'Class_2', 'Class_score',
    'Order', 'Order_2', 'Order_score',
    'Family', 'Family_2', 'Family_score',
    'Genus', 'Genus_2', 'Genus_score']
    rdp = pd.read_table(rdp_path, sep = '\t', index_col=0, names=column_names)

    # Find the most detailed phylogeny that is better than your cutoffs
    phylo_assignment = ''
    phylo_scoring = ['Kingdom_score', 'Phylum_score', 'Class_score', 'Order_score', 
    'Family_score', 'Genus_score']
    for phylo_score in phylo_scoring:
        rdp_score =  rdp.loc[otu].loc[phylo_score]
        if rdp_score >= cutoff:
            # Yay, you found a thing!
            phylo_name = phylo_score.replace('_score','')
            phylo_assignment = rdp.loc[otu].loc[phylo_name]
            #print 'yaaaaay', rdp_score, phylo_assignment
    return phylo_assignment
