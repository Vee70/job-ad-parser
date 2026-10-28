# Job Ad Parser and Skill Matcher

Parses plain text job ads and links them to the ESCO skills taxonomy. It uses spaCy pattern matching to extract job titles and locations, then maps each job title to the closest ESCO label based on word overlap. It also uses sentence embeddings from all-MiniLM-L12-v2 and cosine similarity to find the top-k ESCO descriptions that best match a job title.

*Note: This is an old project. I'm going back through the code to check for issues, so parts of it may change.*
