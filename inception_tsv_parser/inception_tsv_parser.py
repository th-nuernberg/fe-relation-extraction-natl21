import os
import json
import argparse

class Relation():
    def __init__(self, entity_1_id, relation_name, entity_2_id):
        self.entity_1_id = entity_1_id
        self.relation_name = relation_name
        self.entity_2_id = entity_2_id

class InceptionToken():
    def __init__(self, id, position, text, entity_name):
        self.id = id
        self.position = position
        self.text = text
        self.entity_name = entity_name

def find_entities(entity_tokens):
    compound_entities = dict()

    for token in entity_tokens:
        if token.entity_name not in compound_entities.keys():
            compound_entities[token.entity_name] = InceptionToken(token.id, token.position, token.text, token.entity_name)
        else:
            if (token.position[0] < compound_entities[token.entity_name].position[0] and (token.position[1]+1==compound_entities[token.entity_name].position[0] or token.position[1]==compound_entities[token.entity_name].position[0])):
                compound_entities[token.entity_name].position = (token.position[0], compound_entities[token.entity_name].position[1])
                compound_entities[token.entity_name].text = " ".join([token.text, compound_entities[token.entity_name].text])

            if (token.position[1] > compound_entities[token.entity_name].position[1] and (token.position[0]==compound_entities[token.entity_name].position[1]+1 or token.position[0]==compound_entities[token.entity_name].position[1])):
                compound_entities[token.entity_name].position = (compound_entities[token.entity_name].position[0], token.position[1])
                compound_entities[token.entity_name].text = " ".join([compound_entities[token.entity_name].text, token.text])

    token_id2entity_map = dict()

    for entity_name in compound_entities.keys():
        entity = compound_entities[entity_name]
        token_id2entity_map[entity.id] = entity

    for token in entity_tokens:
        if (token.id not in token_id2entity_map):
            token_id2entity_map[token.id] = token

    return token_id2entity_map

def load_entitiy2wikidataIdMapping(path):
    ent_map = dict()
    with open(path, 'r', encoding='UTF-8') as fin:
        for line in fin:
            name, qid = line.strip().split("\t")
            ent_map[name] = qid
    return ent_map

def process_file(inputfile):
    inception_sentences = dict()
    line_number = 0
    sentence_start_index = 0
    next_line_contains_sentence_start_index = False

    #? Parsing inputfile for defined entities and relations
    #? Expected input format looks like this (WebAnno TSV 3.2):
    #? <token_id>	<position>	<text>	<entity_type>	<relation_present>	<relation_type>|<another relation_type>	<relation_partner_token_id>|<another relation_partner_token_id>
    #? Examples:
    #? 2-27	265-277	Indianapolis	Location[7]	A uses/employs charging technology B[9]	A operates \[something\] in location B|A operates \[something\] in location B	2-24[6_7]|2-3[3_7]
    #? 4-16	459-467	Proterra	_	NOTA[11]|A researches/develops technology or product B[12]	_	_

    with open(inputfile, encoding='UTF_8') as in_file:
        substract = 0
        last_token_end=-1
        last_token_start=0
        ignore_block=False
        ignore_block_count=0

        for line in in_file:
            #? Lines starting with '#Text' are marking a new sentence
            if line.startswith('#Text'):
                line_number += 1

                inception_sentences[line_number] = dict()
                inception_sentences[line_number]['sentence'] = line[6:].replace("  "," ")
                inception_sentences[line_number]['entity_tokens'] = []
                inception_sentences[line_number]['relations'] = []

                next_line_contains_sentence_start_index = True
                substract=0

                #? ignore blocks with sentences containing multiple lines
                if ('\\r' in line):
                    ignore_block=True

                if (ignore_block):
                    ignore_block_count+=1
                    if (ignore_block_count>2):
                        ignore_block=False
                        ignore_block_count=0

            elif (not line.startswith('#') and not line.strip() == ""):
                if (not ignore_block):
                    #? Extracting available information from each valid token line in inputfile and storing the found token information
                    columns = line.split("\t")

                    if (next_line_contains_sentence_start_index):
                        sentence_start_index = int(columns[1].split('-')[0])
                        next_line_contains_sentence_start_index = False

                    if (last_token_end == columns[1].split('-')[0] or (last_token_start!= 0 and columns[1].split('-')[0] != 0 and str(int(last_token_end)+1) == columns[1].split('-')[0])):
                        substract -= 1

                    if (not columns[3] == "_"):
                        token_id = columns[0]

                        pos_split = columns[1].split('-')
                        position = ((int(pos_split[0])-sentence_start_index)-substract, (int(pos_split[1])-sentence_start_index)-substract)
                        token_text = columns[2]
                        entity_name = columns[3]

                        if (columns[5] != '_'):
                            rel_name_split = columns[5].split('|')
                            rel_partner_split = columns[6].split('|')

                            for i, rel_name in enumerate(rel_name_split):
                                if ('[' in rel_partner_split[i]):
                                    rel_partner = (rel_partner_split[i])[:rel_partner_split[i].find('[')]
                                else:
                                    rel_partner = rel_partner_split[i]
                                inception_sentences[line_number]['relations'].append(Relation(rel_partner, rel_name, token_id))

                        inception_sentences[line_number]['entity_tokens'].append(InceptionToken(token_id, position, token_text,entity_name))
                        inception_sentences[line_number]['relation_present'] = columns[4]

                    last_token_start = columns[1].split('-')[0]
                    last_token_end = columns[1].split('-')[1]
                    substract += 1

    return inception_sentences

def process_file_old_format(inputfile):
    inception_sentences = dict()
    line_number = 0
    sentence_start_index = 0
    next_line_contains_sentence_start_index = False

    #? Parsing inputfile for defined entities and relations
    #? Expected input format looks like this:
    #? <token_id>	<position>	<text>	<entity_type>	<relation_present>	<relation_type>|<another relation_type>	<relation_partner_token_id>|<another relation_partner_token_id>
    #? Examples:
    #? 2-27	265-277	Indianapolis	Location[7]	A uses/employs charging technology B[9]	A operates \[something\] in location B|A operates \[something\] in location B	2-24[6_7]|2-3[3_7]
    #? 4-16	459-467	Proterra	_	NOTA[11]|A researches/develops technology or product B[12]	_	_

    with open(inputfile, encoding='UTF_8') as in_file:
        for line in in_file:
            #? Lines starting with '#Text' are marking a new sentence
            if line.startswith('#Text'):
                line_number += 1

                inception_sentences[line_number] = dict()
                inception_sentences[line_number]['sentence'] = line[6:]
                inception_sentences[line_number]['entity_tokens'] = []
                inception_sentences[line_number]['relations'] = []

                next_line_contains_sentence_start_index = True
            elif (not line.startswith('#') and not line.strip() == ""):
                #? Extracting available information from each valid token line in inputfile and storing the found token information
                columns = line.split("\t")

                if (next_line_contains_sentence_start_index):
                    sentence_start_index = int(columns[1].split('-')[0])
                    next_line_contains_sentence_start_index = False

                if (not columns[3] == "_"):
                    token_id = columns[0]

                    pos_split = columns[1].split('-')
                    position = (int(pos_split[0])-sentence_start_index, int(pos_split[1])-sentence_start_index)

                    token_text = columns[2]
                    entity_name = columns[3]

                    if (columns[5] != '_'):
                        rel_name_split = columns[5].split('|')
                        rel_partner_split = columns[6].split('|')

                        for i, rel_name in enumerate(rel_name_split):
                            if ('[' in rel_partner_split[i]):
                                rel_partner = (rel_partner_split[i])[:rel_partner_split[i].find('[')]
                            else:
                                rel_partner = rel_partner_split[i]
                            inception_sentences[line_number]['relations'].append(Relation(rel_partner, rel_name, token_id))

                    inception_sentences[line_number]['entity_tokens'].append(InceptionToken(token_id, position, token_text,entity_name))
                    inception_sentences[line_number]['relation_present'] = columns[4]

    return inception_sentences

#? Parsing command line arguments
parser = argparse.ArgumentParser()

parser.add_argument("-i", "--inputdir", default=None, type=str, required=True,
                    help="Directory with tsv files from Inception containing extracted data.")
parser.add_argument("-o", "--outputdir", default=None, type=str, required=False,
                    help="Directory to write output data to.")
parser.add_argument("-e", "--entity_map", default='../ERNIE/kg_embed/entity_map.txt', type=str, required=False,
                    help="The file containing the entity name to wikidata identifier mapping.")
parser.add_argument("-l", "--legacy", action="store_true",
                    help="If set legacy format is processed.")

args = parser.parse_args()
input_dir = args.inputdir
if (args.outputdir):
    output_dir = args.outputdir
else:
    output_dir = args.inputdir
ent_map_file = args.entity_map
use_old_format = args.legacy

for inputfile in [input_dir + os.path.sep + f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]:
    filename = os.path.splitext(os.path.basename(inputfile))[0]
    outputfile = output_dir + os.path.sep + filename + '.json'

    print("Processing file '%s'" % inputfile)

    if (use_old_format):
        inception_sentences = process_file_old_format(inputfile)
    else:
        inception_sentences = process_file(inputfile)

    #? For usage with ERNIE model the training data has to contain wikidata identifiers instead of the entity strings
    #? Therefore the ent_map from ERNIE has to be loaded and the entities have to be converted.
    # ent_map = load_entitiy2wikidataIdMapping(ent_map_file)
    # wikidata_id = ent_map[ent1.text]

    examples = []

    for index in inception_sentences:
        sentence = inception_sentences[index]

        #? Aggregating all tokens of an entity together to one entity element
        token_id2entity_map = find_entities(sentence['entity_tokens'])

        #? Adding an example for each relation found in the inputfile
        for relation in sentence['relations']:
            entities = []

            ent1 = token_id2entity_map[relation.entity_1_id]
            entities.append([ent1.text, ent1.position[0], ent1.position[1], 0])

            ent2 = token_id2entity_map[relation.entity_2_id]
            entities.append([ent2.text, ent2.position[0], ent2.position[1], 0])

            example = dict()
            example['label'] = relation.relation_name
            example['text'] = sentence['sentence'].strip()
            example['ents'] = entities

            examples.append(example)

    #? Writing examples to output file
    os.makedirs(os.path.dirname(outputfile), exist_ok=True)
    with open(outputfile, "w") as out_file:
        json.dump(examples, out_file)