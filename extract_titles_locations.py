import pandas as pd
import spacy
from spacy.matcher import Matcher


def return_titles(fname):

    nlp = spacy.load('en_core_web_sm')

    # load skill ontology file into a DataFrame
    filename = 'skillGroups_en'
    csv_file_path = f'{filename}.csv'

    # read job ads and process the text using `nlp`
    with open(fname, 'r') as f:
        doc_ads = nlp(f.read())

    ### get sentence that starts with "JOB TITLE: "
    matcher_title = Matcher(nlp.vocab)
    pattern_title = [
        {'LOWER': 'job'},
        {'LOWER': 'title'},
        {'IS_PUNCT': True},
        {'IS_SPACE': True, 'OP': '+'},
        {'IS_SPACE': False, 'OP': '+'}
    ]
    # get the longest match
    matcher_title.add('Title', [pattern_title], greedy='LONGEST')
    matches_title = matcher_title(doc_ads)

    job_titles = []
    for _, start, end in matches_title:
        # remove "JOB TITLE: " from the text
        titles = doc_ads[start:end].text.split('JOB TITLE:')[1]
        # splite the title if it contain "|"
        job_titles.extend(
            [title.strip() for title in titles.split('|')]
        )

    ### get match for each title
    df_ontology = pd.read_csv(csv_file_path)
    # get unique skills from ontology
    skills_list = df_ontology['preferredLabel'].tolist()
    
    best_matches = []
    for title in job_titles:
        doc_title = nlp(title)
        # dict to store all matches
        #   key: index of "preferedLabel" in ontology
        #   values: matche output
        # the number of matches is obtained by using len(values)
        matches_dict = {}
        for i, skill in enumerate(skills_list):
            matcher_skill = Matcher(nlp.vocab)
            # add skill specific pattern
            # exclude conjuctions and non alphabetical characters
            pattern_skill = [
                {'LOWER': tok.text.lower(), 'OP': '?'} for tok in nlp(skill)
                    if tok.pos_ != 'CCONJ' and tok.is_alpha
            ]
            matcher_skill.add(f'Skill_{i}', [pattern_skill])
            matches_skill = matcher_skill(doc_title)
            # if there's a match, add to the dict
            if matches_skill and matches_skill not in matches_dict.values():
                matches_dict[i] = matches_skill
        # get all skill indices with largest number of matches
        # if there are more than 1 key (skill) with max number of match
        #  select the one with the shortest len (smallest number of token)
        #  because it less likely to contain irrelevant information
        #   (for example, job title, pucntuation, or other random words)
        max_len = max(len(val) for val in matches_dict.values())
        skill_indices = [idx for idx, val in matches_dict.items()
                             if len(val) == max_len]
        # get all skills with the largest matches
        matches_skill_dict = {}
        for idx in skill_indices:
            matches_skill_dict[idx] = len(nlp.make_doc(skills_list[idx]))
        # select one with the smallest number of tokens
        best_match_idx = min(matches_skill_dict, key=matches_skill_dict.get)
        best_matches.append(skills_list[best_match_idx])

    return '\n'.join(best_matches)

def return_job_locs(fname):

    nlp = spacy.load('en_core_web_sm')

    # read job ads file
    with open(fname, 'r') as f:
        job_ads = f.read()
    doc = nlp(job_ads)

    matcher = Matcher(nlp.vocab)
    # define patterns to extract locations
    # `ent_type_` of city name is `PERSON`
    #  --> use `PROPN` instead of 'GPE'
    pattern = [
        {'POS': 'PROPN'},    # city name
        {'IS_PUNCT': True},  # ,
        {'ENT_TYPE': 'GPE'}, # country name
    ]
    matcher.add('Location', [pattern])
    matches = matcher(doc)

    job_locs = ''
    for _, start, end in matches:
        job_locs += doc[start:end].text + '\n'

    return job_locs
