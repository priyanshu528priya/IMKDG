import selection.database_query as query
import generate.medllama as medllama
import umls.entity_extract as ummls
import selection.exp_meddialog as exp

import json
import chromadb

# Specify the path to your JSON file
json_file_path = path_to_dataset_file_in_json

# ids = []
# documents=[]
# goldResponse = []
# ex1 = []
# ex1Response = []
# ex2 = []
# ex2Response = []
# ex3 = []
# ex3Response = []
# kt = []
# normalResponse = []
# responseWithEx = []
# responseWithKT = []
# responseWithExKT = []

def getNormalResponsePrompt(document):
    return f"""Generate the Doctor's response for the following patient doctor conversation
    {document}"""

def getResponseWithExPrompt(document, examples):
    prompt = """Based on the following examples of doctor's response for patient doctor conversation \n"""

    for i, example in enumerate(examples):
        prompt += f"""Example {i+1} ->
{example}
"""
        
    prompt += f"""Generate the Doctor's response for the following patient doctor conversation
{document}
"""

    return prompt


def getKTResponsePrompt(document, kt):
    return f"""Generate the Doctor's response for the following patient doctor conversation according to the following knowledge triples
{document}
Knowledge Triples: {kt}"""

def getResponseWithExKTPrompt(document, kt, examplesWithKT):
    prompt = """Based on the following examples of doctor's response and following knowledge triples for patient doctor conversation \n"""

    for i, example in enumerate(examplesWithKT):
        prompt += f"""Example {i+1} ->
{example}
"""
        
    prompt += f"""Generate the Doctor's response for the following patient doctor conversation
{document}
Knowledge Triples: {kt}
"""

    return prompt
    

import csv
csv_file_path = 'meddialog-response.csv'


# Open the file in read mode
with open(json_file_path, 'r') as json_file, open(csv_file_path, 'a', newline='') as csvfile:
    # Load the JSON data from the file
    data = json.load(json_file)
    csv_writer = csv.writer(csvfile, delimiter='|')

    idx = 1

    for json_object in data:
        if len(json_object['utterances'])>1:
            
            id = "id"+str(idx)
            document = json_object['utterances'][0]
            kt = ummls.extractRelations(document)
            goldResponse = json_object['utterances'][1]
            idx+=1

            matcher = exp.Matcher(5)
            results = matcher.getMatchingExemplars(json_object['utterances'][0], 3)
            
            ex1, ex2, ex3 = (results[0][0], results[0][1], results[0][2])
            kts = (str(ummls.extractRelations(ex1)), str(ummls.extractRelations(ex2)), str(ummls.extractRelations(ex3)))
            ex1Response, ex2Response, ex3Response = (results[1][0], results[1][1], results[1][2])

            examples = []
            examplesWithKT = []
            for i in range(3):
                examples.append(results[0][i]+"\n"+results[1][i])
                examplesWithKT.append(results[0][i]+"\nKnowledge Triples: "+kts[i]+"\n" + results[1][i])

            
            # print(f"Normal Response Prompt: {getNormalResponsePrompt(document)}")
            # print(f"Response with Ex Prompt: {getResponseWithExPrompt(document, examples)}")
            # print(f"Response with KT Prompt: {getKTResponsePrompt(document, kt)}")
            # print(f"Response with Ex and KT Prompt: {getResponseWithExKTPrompt(document, kt, examplesWithKT)}")

            print("I am starting)")
            normalResponse = medllama.responseGeneration(getNormalResponsePrompt(document))
            print("Normal Response done")
            responseWithEx = medllama.responseGeneration(getResponseWithExPrompt(document, examples))
            print("Response with Ex done")
            responseWithKT = medllama.responseGeneration(getKTResponsePrompt(document, kt))
            print("Response with KT done")
            responseWithExKT = medllama.responseGeneration(getResponseWithExKTPrompt(document, kt, examplesWithKT))
            print("Response with Ex and KT done")

            new_row_data = [id, goldResponse, normalResponse, responseWithEx, responseWithKT, responseWithExKT]

            csv_writer.writerow(new_row_data)
            print(f"Row {id} done")
                        
