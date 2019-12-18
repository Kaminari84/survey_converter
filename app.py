from flask import Flask, request, make_response, render_template, current_app, g
import json
import os
import random
import time
import numpy as np
import requests
from urllib.parse import quote
from bs4 import BeautifulSoup
import spacy

from probReactionChoices import probReactionChoices
from reactions import reaction_repo

from m_classicTextClf import TextClassifier


app = Flask(__name__)
app.app_context().push()


def initReactionRepo():
    for key, value in reaction_repo.items():
        mgr = probReactionChoices(value)
        current_app.reactionMgrs[key] = mgr

def setup_app(app):
    print("Loading the server, first init global vars...")
    #DataMgr.load_team_auths()
    #logging.info('Creating all database tables...')
    #db.create_all()
    #logging.info('Done!')

    #print("ENVIRON TEST:", os.environ['TEST_VAR'])

    with app.app_context():
        # within this block, current_app points to app.
        print("App name:",current_app.name)
        current_app.reactionMgrs = {}

        initReactionRepo()

        #POS tagger
        current_app.nlp = spacy.load("en_core_web_sm")
        current_app.nlp.vocab["indicate"].is_stop = True
        current_app.nlp.vocab["extent"].is_stop = True

        #train question framing classifier
        current_app.q_clf = TextClassifier("./static/model_data/q_framing_data.json")
        current_app.q_clf.train(test_prop=0.2)

        #train answer framing classifier
        current_app.ans_clf = TextClassifier("./static/model_data/ans_types.json")
        current_app.ans_clf.train(test_prop=0.1)

        #train survey domain recongition classifier
        current_app.d_clf = TextClassifier("./static/model_data/domain_survey_data.json")
        current_app.d_clf.train(test_prop=0.2, rem_stop_words=True)

    print("Start the actual server...")

setup_app(app)

# Server paths
@app.route('/')
def hello_world():
    return 'Hello, World!'

@app.route('/conv_survey')
def conv_survey():
    survey_files = os.listdir('./static/surveys/')

    return render_template('bot_interface.html', surveys = survey_files )

@app.route('/get_survey_list')
def get_survey_list():
    print('Get survey list called:')

    survey_files = os.listdir('./static/surveys/')
    json_resp = json.dumps({'status': 'OK', 'message':'', 'surveys':survey_files})

    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route("/get_survey")
def get_survey():
    print("Trying to get survey..")

    survey_file = request.args.get('survey_file')
    print("Survey file:",str(survey_file))

    survey_dict = ""
    with open('./static/surveys/'+survey_file, 'r') as f:
        survey_dict = json.load(f)

    json_resp = json.dumps({'status': 'OK', 'message':'', 'survey_data':survey_dict})

    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route("/scrape_survey")
def scrape_survey():
    print("Trying to scrape a survey..")

    survey_url = request.args.get('survey_url')
    survey_name = survey_url .rsplit('/', 1)[-1].replace(" ","_")
    print("Survey URL:",str(survey_url), ", Survey name:",survey_name)

    question_list = scrapeSurveyGizmo(survey_url)
    with open('./static/surveys/'+survey_name+".json", 'w') as f:
        #json.dump(question_list, f)
        str_json = json.dumps(question_list, indent=4, sort_keys=True)
        str_json = str_json.replace("'","\\'")
        str_json = str_json.replace("\'","\\'")
        print(str_json, file=f)

    json_resp = json.dumps({'status': 'OK', 
                            'message':'', 
                            'survey_file': survey_name+".json",
                            'survey_data': question_list})

    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route("/q_framing", methods = ['GET'])
def q_framing():
    print("Getting framing for question...")

    q = request.args.get('q')
    print("q=",str(q))

    q_frame = callQuestionFraming(q)
    print("Frame:"+q_frame)

    json_resp = json.dumps({'status': 'OK', 
                            'message':'', 
                            'q': q,
                            'q_framing':q_frame})

    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route("/ans_framing", methods = ['GET'])
def ans_framing():
    print("Getting framing for answer options...")

    ans = request.args.get('a')
    print("a=",str(ans))

    ans_frame = current_app.ans_clf.classify([ans])[0]
    print("Frame:"+ans_frame)

    json_resp = json.dumps({'status': 'OK', 
                            'message':'', 
                            'a': ans,
                            'ans_framing':ans_frame})

    return make_response(json_resp, 200, {"content_type":"application/json"})

@app.route("/augment_survey", methods = ['POST'])
def augment_survey():
    print("Trying to agument the survey...")

    survey_file = request.form.get('survey_file')
    print("Survey file:",str(survey_file))

    survey_questions = ""
    with open('./static/surveys/'+survey_file, 'r') as f:
        survey_questions = json.load(f)

    #survey_content = request.form.get('survey_content')
    #print("Survey content:",str(survey_content))
    #survey_questions = json.loads(survey_content)

    # Augmentation params
    isOpening = True if request.form.get('isOpening') == "true" else False # Is intro added?
    isClosing = True if request.form.get('isClosing') == "true" else False # Is closing added?

    print("IS OPENING:", isOpening)
    print("IS CLOSING:", isClosing)

    progress_repeat = int(request.form.get('progressRepeatN')) # Every how many questions does the progress reporting repeat
    reaction_repeat = int(request.form.get('reactionRepeatN')) # Every how many questions does the reaction happen

    print("REACTION REPEAT EVERY:", reaction_repeat)
    print("PROGRESS REPEAT EVERY:", progress_repeat)

    # Empathy: 1.0 - all reactins are empathy loaded, 0.0 - all reactions are neutral
    reaction_empathy = float(request.form.get('empathyLevel')) 
    print("EMPATHY LEVEL:", reaction_empathy)

    # Q Augmentation: 1.0 - all question augmented, 0.0 - none augmented
    q_augment = float(request.form.get('qAugmentLevel')) 
    print("QUESTION AUGMENTATION LEVEL:", q_augment)

    print("-------ORIGINAL--------")
    # Show the original survey
    i = 0
    all_survey_words = ""
    for question in survey_questions:
        i += 1
        print(str(i)+".", question['text'], ", A: ", question['type'])
        #rem stop words
        sent_tokens = current_app.nlp(question['text'].lower())
        for token in sent_tokens:
            if token.is_stop:
                pass
            else:
                all_survey_words += token.text+" "
    
    print("DOMAIN ["+str(len(all_survey_words))+"]->", all_survey_words)
    # Recognize the survey domain
    topic = "unknown"
    with app.app_context():
        topic = str(current_app.d_clf.classify([all_survey_words])[0])

    print("Detected domain:", topic)
    
    # Deep content params
    name = str(topic).title().replace(" ","")+"Bot"
    print("Bot name:", name)
    sections = ["Habitual Action", "Understanding", "Reflection", "Critical Reflection"]

    #Add survey progress blocks
    print("-------PROCESSING-------")
    i = 0
    with app.app_context():
        for question in survey_questions:
            if "//" in question: 
                print("Comment found")
            #Progress
            if progress_repeat > 0 and i % progress_repeat == (progress_repeat-1):
                pos = round(i*10//len(survey_questions)*10)
                progress_text = current_app.reactionMgrs['progress_list'].getLeastFreqChoice()  #random.choice(progress_list)
                if (pos > 40 and pos < 60):
                    progress_text = current_app.reactionMgrs['progress_middle_list'].getLeastFreqChoice() #random.choice(progress_middle_list)
                elif (pos > 85):
                    progress_text = current_app.reactionMgrs['progress_end_list'].getLeastFreqChoice() #random.choice(progress_end_list)

                block = {}
                block['text'] = progress_text.replace("{d}", str(i)).replace("{n}",str(len(survey_questions))).replace("{l}", str(len(survey_questions)-i)).replace("{percent}",str(round(i*10//len(survey_questions)*10)))
                block['type'] = "Skip"
                block['source'] = "augmented"

                survey_questions.insert(i, block)
 
            #Reactions
            if reaction_repeat > 0 and i % reaction_repeat == (reaction_repeat-1):
                react_empathicly = (random.random() <= reaction_empathy)
                #print("Rand:", react_empathicly)
                #print("Q:",question)

                reaction_block = {}

                #Add empathetic reaction that matches the context (question-answer)
                if react_empathicly:
                    q_frame = callQuestionFraming(question['text'])
                    print("Frame:"+q_frame)
                    question['framing'] = q_frame

                    # For Yes/No type
                    if question['type'] == 'Yes/No':
                        positive_react = current_app.reactionMgrs['reaction_positive_list'].getLeastFreqChoice() #random.choice(reaction_positive_list)
                        negative_react = current_app.reactionMgrs['reaction_negative_list'].getLeastFreqChoice() #random.choice(reaction_negative_list)
                        
                        if q_frame == "Positive_frame":
                            reaction_block['Yes'] = positive_react
                            reaction_block['No'] = negative_react
                        elif q_frame == "Negative_frame":
                            reaction_block['Yes'] = negative_react
                            reaction_block['No'] = positive_react
                    # For Options
                    if question['type'] == 'Options':
                        react = current_app.reactionMgrs['reaction_neutral_list'].getLeastFreqChoice() #random.choice(reaction_neutral_list)
                        print("Orig neutr react:", react)
                        for option in question['options']:
                            print("Option", option['text'])

                            ans_frame = current_app.ans_clf.classify([option['text']])[0]
                            option['framing'] = ans_frame
                            
                            # Positive Answers
                            #if option['text'] in pos_answers:
                            if ans_frame == "pos_answer":
                                if q_frame == "Positive_frame":
                                    react = current_app.reactionMgrs['reaction_positive_list'].getLeastFreqChoice() #random.choice(reaction_positive_list)
                                elif q_frame == "Negative_frame":
                                    react = current_app.reactionMgrs['reaction_negative_list'].getLeastFreqChoice() #random.choice(reaction_negative_list)
                            # Negative Answers
                            #elif option['text'] in neg_answers:
                            if ans_frame == "neg_answer":
                                if q_frame == "Positive_frame":
                                    react = current_app.reactionMgrs['reaction_negative_list'].getLeastFreqChoice() #random.choice(reaction_negative_list)
                                elif q_frame == "Negative_frame":
                                    react = current_app.reactionMgrs['reaction_positive_list'].getLeastFreqChoice() #random.choice(reaction_positive_list)
                            # Neutral Answers or Unknown
                            else:
                                pass
                                
                            #Assign reaction to option
                            reaction_block[option['value']] = react
                        
                else:
                    # For Yes/No type
                    if question['type'] == 'Yes/No':
                        neutral_react =  current_app.reactionMgrs['reaction_neutral_list'].getLeastFreqChoice() #random.choice(reaction_neutral_list)
                        reaction_block['Yes'] = neutral_react
                        reaction_block['No'] = neutral_react
                    # For Options
                    if question['type'] == 'Options':
                        react = current_app.reactionMgrs['reaction_neutral_list'].getLeastFreqChoice() #random.choice(reaction_neutral_list)
                        for option in question['options']:
                            reaction_block[option['value']] = react
                        
                
                if bool(reaction_block):
                    question['reactions'] = reaction_block

            #Question augmentation
            if q_augment > 0:
                addPrefix = (random.random() <= q_augment)
                if addPrefix:
                    prefix = ""
                    if i==0:
                        prefix = current_app.reactionMgrs['q_prefix_start_list'].getLeastFreqChoice()
                    else:
                        prefix = current_app.reactionMgrs['q_prefix_list'].getLeastFreqChoice()

                    q_tokens = current_app.nlp(question['text'])

                    #print("Question tokens:")
                    #for token in q_tokens:
                    #    print(token.text+" -> "+token.pos_)

                    question['org_text'] = question['text']

                    question['prefix'] = prefix
                    if q_tokens[0].pos_ == "VERB":
                        print("First token -> verb")
                        question['mod_text'] = question['text'].replace(q_tokens[0].text+" ","")
                    else:
                        question['mod_text'] = question['text'][0].lower()+question['text'][1:]

                    question['text'] = prefix+" "+question['text'][0].lower()+question['text'][1:]

            i += 1

        #Add beginning
        if (isOpening):
            print("Adding OPENING block")
            block = {}
            close_text = current_app.reactionMgrs['intro_list'].getLeastFreqChoice() #random.choice(intro_list)
            block['text'] = close_text.replace("{name}",name).replace("{topic}",topic)
            block['type'] = "Skip"
            block['source'] = "augmented"

            survey_questions.insert(0, block)

        #Add ending
        if (isClosing):
            print("Adding CLOSING block")
            block = {}
            close_text = current_app.reactionMgrs['close_list'].getLeastFreqChoice() #random.choice(close_list)
            block['text'] = close_text
            block['type'] = "End"
            block['source'] = "augmented"

            survey_questions.append(block)

    #Survey after all the changes
    print("-------AFTER CHANGES--------")
    i = 0
    for question in survey_questions:
        i += 1
        print(str(i)+".", question['text'])
        print("\tA: ", question['type'])
        if ('reactions' in question):
            print("\tR:", end='')
            for reaction_key in question['reactions']:
                print(" | ",reaction_key,":",question['reactions'][reaction_key], end='')
            print()


        

    #print("Trimmed:", survey_dict[0:1])

    #Save to JS file
    #with open('./static/conv_surveys/conv_survey.js', 'w') as f:
    #    str_json = json.dumps(survey_questions)
    #    str_json = str_json.replace("'","\\'")
    #    str_json = str_json.replace("\'","\\'")
    #    print("bot_name='"+name+"';", file=f)
    #    print("data='"+str_json+"'", file=f)

    #Save to json file
    conv_survey_file = survey_file.replace('.json','_conv.json')
    print("Saving conversational survey file:",conv_survey_file)
    with open('./static/conv_surveys/'+conv_survey_file, 'w') as f:
        str_json = json.dumps(survey_questions, indent=4, sort_keys=True)
        str_json = str_json.replace("'","\\'")
        str_json = str_json.replace("\'","\\'")
        print(str_json, file=f)

    
    cov_survey = survey_questions
    json_resp = json.dumps({'status': 'OK', 'message':'', 
                            'survey_data':cov_survey,
                            'conv_survey_file': conv_survey_file,
                            'domain':topic, 'bot_name':name})

    return make_response(json_resp, 200, {"content_type":"application/json"})



#Survey Gizmo survey scraping code
def scrapeSurveyGizmo(url):
    api_url_base = url+'?__sgtarget='

    questions = []
     
    for page in range(1,8):
        final_url = api_url_base+str(page)
        print("PAGE:", final_url)
        response = requests.get(final_url, timeout=5)

        if response.status_code == 200:
            #print("Rresp:",response.content.decode('utf-8'))
            page_content = BeautifulSoup(response.content, "html.parser")
            
            #scrape the questions
            divs = page_content.find_all("", {"class": "sg-question"})
            print("Q Num:", len(divs))
            for elem in divs:
                ans_type = None
                if "sg-type-radio" in elem['class']:
                    ans_type = "Options"
                elif "sg-type-textbox" in elem['class']:
                    ans_type = "Input"

                q_title = elem.find("", {"class": "sg-question-title"})
                if q_title:
                    q_text = None
                    if (q_title.find("label")):
                        q_text = str(q_title.label.text).strip() #str(q_title.label.contents[2]).strip()
                    else:
                        q_text = str(q_title.text).strip()
                    if (q_text):
                        q_text = q_text.replace('*This question is required.','')[3:]

                #get the option labels
                ans_options = []
                if ans_type == "Options":
                    q_options = elem.find("div", {"class": "sg-question-options"})
                    q_opts_elems = q_options.find_all("input", {"class": "sg-input"})
                    opt_n = 0
                    for opt in q_opts_elems:
                        opt_n+=1
                        ans_options.append({"value":"opt_"+str(opt_n), 
                                            "text":opt['aria-label']})
                        #print("\tLabel:",opt['aria-label'])
                    

                if ans_type != None and q_text != None:
                    questions.append({"url":final_url,
                                     "type": ans_type, 
                                      "text": q_text,
                                      "options": ans_options})
        else:
            print("ERROR:"+str(response.text))
    
    #Go through all the questions
    print("--- Total Questions: "+str(len(questions))+" ---")
    for q in questions:
        print("Q:"+q['text'])
        print("\t"+q['type']+":",q['options'])
    
    return questions
    
#Call LUIS to classify the questions framing
def callQuestionFraming(q, model="classic"):
    if model == "classic":
        labels = current_app.q_clf.classify([q])
        return labels[0]

    elif model == "rasa":
        return None
    elif model == "py_porch":
        return None
    elif model == "luis":
        api_key = 'bc63aae2cc1b4ab8977765baaf450696'
        api_url_base = 'https://westus.api.cognitive.microsoft.com/luis/v2.0/apps/'
        app_id = '1c7de8fb-8ae0-4b46-ba1b-f1c16088c6da'
        url_q = quote(q)

        #print("Quoted string:"+url_q)
        print("Question:"+q)

        time.sleep(0.300) #300 milliseconds

        call = api_url_base+""+app_id+"?verbose=true&timezoneOffset=0&subscription-key="+api_key+"&q="+url_q
        #print(call)

        response = requests.get(call)

        if response.status_code == 200:
            json_resp = json.loads(response.content.decode('utf-8'))
            return json_resp['topScoringIntent']['intent']
        else:
            print("ERROR:"+str(response.text))
            return None