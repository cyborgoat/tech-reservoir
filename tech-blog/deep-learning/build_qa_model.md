---
title: Build a simple Question Answering model for SQuAD dataset
summary: This article introduces SQuAD dataset and how to use this dataset to perform a simple question answering system with BERT.
author: Junxiao Guo
date: 2023-03-26
tags:
  - nlp
  - deep-learning
  - question-answering
---

## SQuAD 2.0 Dataset

The SQuAD (Stanford Question and Answering Dataset) is a hugely popular dataset containing question and answer pairs scraped from Wikipedia, covering topics ranging from Beyonce, to Physics. As one of the most comprehensive Q&A datasets available, it's only natural that we will be making use of it. So let's explore it.

First, we'll need to download the data. There are two JSON files that we are interested in - train and dev, which we can downloaded from http. Here we will be storing the SQuAD data in the `data/squad` directory, so we must check if this already exists and if not create the directory.

```python
import os

squad_dir = './data/squad'

if not os.path.exists(squad_dir):
    os.mkdir(squad_dir)
```

```python
url = 'https://rajpurkar.github.io/SQuAD-explorer/dataset/'
files = ['train-v2.0.json', 'dev-v2.0.json']
```

```python
# Download data to local machine
import requests

for file in files:
    res = requests.get(url+file)
    # write to file in chunks
    with open(os.path.join(squad_dir, file), 'wb') as f:
        for chunk in res.iter_content(chunk_size=40):
            f.write(chunk)
```

```python

import json

with open(os.path.join(squad_dir, 'train-v2.0.json'), 'rb') as f:
    squad = json.load(f)
```

The JSON structure contains a top-level 'data' key which contains a list of groups, where each group is a topic, such as Beyonce, Chopin, or Matter. We can take a look at the first and last groups respectively.

```python
# squad['data'][0]
```

```python
squad['data'][-1]['paragraphs'][0]
```

    {'qas': [{'plausible_answers': [{'text': 'ordinary matter composed of atoms',
         'answer_start': 50}],
       'question': 'What did the term matter include after the 20th century?',
       'id': '5a7db48670df9f001a87505f',
       'answers': [],
       'is_impossible': True},
      {'plausible_answers': [{'text': 'matter', 'answer_start': 59}],
       'question': 'What are atoms composed of?',
       'id': '5a7db48670df9f001a875060',
       'answers': [],
       'is_impossible': True},
      {'plausible_answers': [{'text': 'light or sound', 'answer_start': 128}],
       'question': 'What are two examples of matter?',
       'id': '5a7db48670df9f001a875061',
       'answers': [],
       'is_impossible': True},
      {'plausible_answers': [{'text': "its (possibly massless) constituents' motion and interaction energies",
         'answer_start': 315}],
       'question': "What can an object's mass not come from?",
       'id': '5a7db48670df9f001a875062',
       'answers': [],
       'is_impossible': True},
      {'plausible_answers': [{'text': 'fundamental', 'answer_start': 449}],
       'question': 'Matter is currently considered to be what kind of concept?',
       'id': '5a7db48670df9f001a875063',
       'answers': [],
       'is_impossible': True}],
     'context': "Before the 20th century, the term matter included ordinary matter composed of atoms and excluded other energy phenomena such as light or sound. This concept of matter may be generalized from atoms to include any objects having mass even when at rest, but this is ill-defined because an object's mass can arise from its (possibly massless) constituents' motion and interaction energies. Thus, matter does not have a universal definition, nor is it a fundamental concept in physics today. Matter is also used loosely as a general term for the substance that makes up all observable physical objects."}

If we compare the first entry on Beyonce and the second on Matter, we can see that we sometimes return our answers in the `answers` key, and sometimes in the `plausible_answers` key. So when processing this data we will need to consider some additional logic to deal with this.

Secondly, for all samples, we need to iterate through multiple levels. On the highest level we have groups, which is where our topics like 'Beyonce' and 'Matter' belong. At the next layer we have paragraphs, and in the next we have our question-answer pairs, this structure looks like this:

We'll work through parsing this data into a cleaner format that we will be using in later notebooks. We need to create a format that consists of a list of dictionaries where each dictionary contains a single question, answer, and context.

```python
# initialize list where we will place all of our data
new_squad = []

# we need to loop through groups -> paragraphs -> qa_pairs
for group in squad['data']:
    for paragraph in group['paragraphs']:
        # we pull out the context from here
        context = paragraph['context']
        for qa_pair in paragraph['qas']:
            # we pull out the question
            question = qa_pair['question']
            # now the logic to check if we have 'answers' or 'plausible_answers'
            if 'answers' in qa_pair.keys() and len(qa_pair['answers']) > 0:
                answer = qa_pair['answers'][0]['text']
            elif 'plausible_answers' in qa_pair.keys() and len(qa_pair['plausible_answers']) > 0:
                answer = qa_pair['plausible_answers'][0]['text']
            else:
                # this shouldn't happen, but just in case we just set answer = None
                answer = None
            # append dictionary sample to parsed squad
            new_squad.append({
                'question': question,
                'answer': answer,
                'context': context
            })
```

```python
new_squad[:2], new_squad[-2:]
```

    ([{'question': 'When did Beyonce start becoming popular?',
       'answer': 'in the late 1990s',
       'context': 'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".'},
      {'question': 'What areas did Beyonce compete in when she was growing up?',
       'answer': 'singing and dancing',
       'context': 'Beyoncé Giselle Knowles-Carter (/biːˈjɒnseɪ/ bee-YON-say) (born September 4, 1981) is an American singer, songwriter, record producer and actress. Born and raised in Houston, Texas, she performed in various singing and dancing competitions as a child, and rose to fame in the late 1990s as lead singer of R&B girl-group Destiny\'s Child. Managed by her father, Mathew Knowles, the group became one of the world\'s best-selling girl groups of all time. Their hiatus saw the release of Beyoncé\'s debut album, Dangerously in Love (2003), which established her as a solo artist worldwide, earned five Grammy Awards and featured the Billboard Hot 100 number-one singles "Crazy in Love" and "Baby Boy".'}],
     [{'question': 'Matter usually does not need to be used in conjunction with what?',
       'answer': 'a specifying modifier',
       'context': 'The term "matter" is used throughout physics in a bewildering variety of contexts: for example, one refers to "condensed matter physics", "elementary matter", "partonic" matter, "dark" matter, "anti"-matter, "strange" matter, and "nuclear" matter. In discussions of matter and antimatter, normal matter has been referred to by Alfvén as koinomatter (Gk. common matter). It is fair to say that in physics, there is no broad consensus as to a general definition of matter, and the term "matter" usually is used in conjunction with a specifying modifier.'},
      {'question': 'What field of study has a variety of unusual contexts?',
       'answer': 'physics',
       'context': 'The term "matter" is used throughout physics in a bewildering variety of contexts: for example, one refers to "condensed matter physics", "elementary matter", "partonic" matter, "dark" matter, "anti"-matter, "strange" matter, and "nuclear" matter. In discussions of matter and antimatter, normal matter has been referred to by Alfvén as koinomatter (Gk. common matter). It is fair to say that in physics, there is no broad consensus as to a general definition of matter, and the term "matter" usually is used in conjunction with a specifying modifier.'}])

```python
with open(os.path.join(squad_dir, 'train.json'), 'w') as f:
    json.dump(new_squad, f)
```

Then we do the same for our development data

```python
import os
import json

with open(os.path.join(squad_dir, 'dev-v2.0.json'), 'rb') as f:
    squad = json.load(f)
```

```python
squad['data'][0]['paragraphs'][0]
```

    {'qas': [{'question': 'In what country is Normandy located?',
       'id': '56ddde6b9a695914005b9628',
       'answers': [{'text': 'France', 'answer_start': 159},
        {'text': 'France', 'answer_start': 159},
        {'text': 'France', 'answer_start': 159},
        {'text': 'France', 'answer_start': 159}],
       'is_impossible': False},
      {'question': 'When were the Normans in Normandy?',
       'id': '56ddde6b9a695914005b9629',
       'answers': [{'text': '10th and 11th centuries', 'answer_start': 94},
        {'text': 'in the 10th and 11th centuries', 'answer_start': 87},
        {'text': '10th and 11th centuries', 'answer_start': 94},
        {'text': '10th and 11th centuries', 'answer_start': 94}],
       'is_impossible': False},
      {'question': 'From which countries did the Norse originate?',
       'id': '56ddde6b9a695914005b962a',
       'answers': [{'text': 'Denmark, Iceland and Norway', 'answer_start': 256},
        {'text': 'Denmark, Iceland and Norway', 'answer_start': 256},
        {'text': 'Denmark, Iceland and Norway', 'answer_start': 256},
        {'text': 'Denmark, Iceland and Norway', 'answer_start': 256}],
       'is_impossible': False},
      {'question': 'Who was the Norse leader?',
       'id': '56ddde6b9a695914005b962b',
       'answers': [{'text': 'Rollo', 'answer_start': 308},
        {'text': 'Rollo', 'answer_start': 308},
        {'text': 'Rollo', 'answer_start': 308},
        {'text': 'Rollo', 'answer_start': 308}],
       'is_impossible': False},
      {'question': 'What century did the Normans first gain their separate identity?',
       'id': '56ddde6b9a695914005b962c',
       'answers': [{'text': '10th century', 'answer_start': 671},
        {'text': 'the first half of the 10th century', 'answer_start': 649},
        {'text': '10th', 'answer_start': 671},
        {'text': '10th', 'answer_start': 671}],
       'is_impossible': False},
      {'plausible_answers': [{'text': 'Normans', 'answer_start': 4}],
       'question': "Who gave their name to Normandy in the 1000's and 1100's",
       'id': '5ad39d53604f3c001a3fe8d1',
       'answers': [],
       'is_impossible': True},
      {'plausible_answers': [{'text': 'Normandy', 'answer_start': 137}],
       'question': 'What is France a region of?',
       'id': '5ad39d53604f3c001a3fe8d2',
       'answers': [],
       'is_impossible': True},
      {'plausible_answers': [{'text': 'Rollo', 'answer_start': 308}],
       'question': 'Who did King Charles III swear fealty to?',
       'id': '5ad39d53604f3c001a3fe8d3',
       'answers': [],
       'is_impossible': True},
      {'plausible_answers': [{'text': '10th century', 'answer_start': 671}],
       'question': 'When did the Frankish identity emerge?',
       'id': '5ad39d53604f3c001a3fe8d4',
       'answers': [],
       'is_impossible': True}],
     'context': 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.'}

```python
squad['data'][-1]['paragraphs'][0]
```

    {'qas': [{'question': 'What concept did philosophers in antiquity use to study simple machines?',
       'id': '573735e8c3c5551400e51e71',
       'answers': [{'text': 'force', 'answer_start': 46},
        {'text': 'force', 'answer_start': 46},
        {'text': 'the concept of force', 'answer_start': 31},
        {'text': 'the concept of force', 'answer_start': 31},
        {'text': 'force', 'answer_start': 46},
        {'text': 'force', 'answer_start': 46}],
       'is_impossible': False},
      {'question': 'What was the belief that maintaining motion required force?',
       'id': '573735e8c3c5551400e51e72',
       'answers': [{'text': 'fundamental error', 'answer_start': 387},
        {'text': 'A fundamental error', 'answer_start': 385},
        {'text': 'A fundamental error', 'answer_start': 385},
        {'text': 'A fundamental error', 'answer_start': 385},
        {'text': 'A fundamental error', 'answer_start': 385},
        {'text': 'A fundamental error', 'answer_start': 385}],
       'is_impossible': False},
      {'question': 'Who had mathmatical insite?',
       'id': '573735e8c3c5551400e51e73',
       'answers': [{'text': 'Sir Isaac Newton', 'answer_start': 654},
        {'text': 'Sir Isaac Newton', 'answer_start': 654},
        {'text': 'Sir Isaac Newton', 'answer_start': 654},
        {'text': 'Sir Isaac Newton', 'answer_start': 654},
        {'text': 'Sir Isaac Newton', 'answer_start': 654},
        {'text': 'Sir Isaac Newton', 'answer_start': 654}],
       'is_impossible': False},
      {'question': "How long did it take to improve on Sir Isaac Newton's laws of motion?",
       'id': '573735e8c3c5551400e51e74',
       'answers': [{'text': 'nearly three hundred years', 'answer_start': 727},
        {'text': 'nearly three hundred years', 'answer_start': 727},
        {'text': 'nearly three hundred years', 'answer_start': 727},
        {'text': 'nearly three hundred years', 'answer_start': 727},
        {'text': 'nearly three hundred years', 'answer_start': 727},
        {'text': 'three hundred years', 'answer_start': 734}],
       'is_impossible': False},
      {'question': 'Who develped the theory of relativity?',
       'id': '573735e8c3c5551400e51e75',
       'answers': [{'text': 'Einstein', 'answer_start': 782},
        {'text': 'Einstein', 'answer_start': 782},
        {'text': 'Einstein', 'answer_start': 782},
        {'text': 'Einstein', 'answer_start': 782},
        {'text': 'Einstein', 'answer_start': 782},
        {'text': 'Einstein', 'answer_start': 782}],
       'is_impossible': False},
      {'plausible_answers': [{'text': 'Philosophers', 'answer_start': 0}],
       'question': 'Who used the concept of antiquity in the study of stationary and moving objects?',
       'id': '5ad25efad7d075001a428f56',
       'answers': [],
       'is_impossible': True},
      {'plausible_answers': [{'text': 'motion', 'answer_start': 377}],
       'question': 'Something that is considered a non fundamental error is the belief that a force is required to maintain what?',
       'id': '5ad25efad7d075001a428f57',
       'answers': [],
       'is_impossible': True},
      {'plausible_answers': [{'text': 'Galileo Galilei and Sir Isaac Newton',
         'answer_start': 585}],
       'question': 'Most of the previous understandings about motion and force were corrected by whom?',
       'id': '5ad25efad7d075001a428f58',
       'answers': [],
       'is_impossible': True},
      {'plausible_answers': [{'text': 'motion and force', 'answer_start': 539}],
       'question': 'Sir Galileo Galilei corrected the previous misunderstandings about what?',
       'id': '5ad25efad7d075001a428f59',
       'answers': [],
       'is_impossible': True},
      {'plausible_answers': [{'text': 'Sir Isaac Newton', 'answer_start': 654}],
       'question': 'Who formulated the laws of motion that were not improved-on for nearly three thousand years?',
       'id': '5ad25efad7d075001a428f5a',
       'answers': [],
       'is_impossible': True}],
     'context': 'Philosophers in antiquity used the concept of force in the study of stationary and moving objects and simple machines, but thinkers such as Aristotle and Archimedes retained fundamental errors in understanding force. In part this was due to an incomplete understanding of the sometimes non-obvious force of friction, and a consequently inadequate view of the nature of natural motion. A fundamental error was the belief that a force is required to maintain motion, even at a constant velocity. Most of the previous misunderstandings about motion and force were eventually corrected by Galileo Galilei and Sir Isaac Newton. With his mathematical insight, Sir Isaac Newton formulated laws of motion that were not improved-on for nearly three hundred years. By the early 20th century, Einstein developed a theory of relativity that correctly predicted the action of forces on objects with increasing momenta near the speed of light, and also provided insight into the forces produced by gravitation and inertia.'}

```python
# initialize list where we will place all of our data
new_squad = []

# we need to loop through groups -> paragraphs -> qa_pairs
for group in squad['data']:
    for paragraph in group['paragraphs']:
        # we pull out the context from here
        context = paragraph['context']
        for qa_pair in paragraph['qas']:
            # we pull out the question
            question = qa_pair['question']
            # now the logic to check if we have 'answers' or 'plausible_answers'
            if 'answers' in qa_pair.keys() and len(qa_pair['answers']) > 0:
                answer_list = qa_pair['answers']
            elif 'plausible_answers' in qa_pair.keys() and len(qa_pair['plausible_answers']) > 0:
                answer_list = qa_pair['plausible_answers']
            else:
                # this shouldn't happen, but just in case we just set answer = []
                answer_list = []
            # we want to pull our the 'text' of each answer in our list of answers
            answer_list = [item['text'] for item in answer_list]
            # we can remove duplicate answers by converting our list to a set, and then back to a list
            answer_list = list(set(answer_list))
            # we iterate through each unique answer in the answer_list
            for answer in answer_list:
                # append dictionary sample to parsed squad
                new_squad.append({
                    'question': question,
                    'answer': answer,
                    'context': context
                })
```

```python
new_squad[:3], new_squad[-2:]
```

    ([{'question': 'In what country is Normandy located?',
       'answer': 'France',
       'context': 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.'},
      {'question': 'When were the Normans in Normandy?',
       'answer': '10th and 11th centuries',
       'context': 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.'},
      {'question': 'When were the Normans in Normandy?',
       'answer': 'in the 10th and 11th centuries',
       'context': 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.'}],
     [{'question': 'What force leads to a commonly used unit of mass?',
       'answer': 'kilogram-force',
       'context': 'The pound-force has a metric counterpart, less commonly used than the newton: the kilogram-force (kgf) (sometimes kilopond), is the force exerted by standard gravity on one kilogram of mass. The kilogram-force leads to an alternate, but rarely used unit of mass: the metric slug (sometimes mug or hyl) is that mass that accelerates at 1 m·s−2 when subjected to a force of 1 kgf. The kilogram-force is not a part of the modern SI system, and is generally deprecated; however it still sees use for some purposes as expressing aircraft weight, jet thrust, bicycle spoke tension, torque wrench settings and engine output torque. Other arcane units of force include the sthène, which is equivalent to 1000 N, and the kip, which is equivalent to 1000 lbf.'},
      {'question': 'What force is part of the modern SI system?',
       'answer': 'kilogram-force',
       'context': 'The pound-force has a metric counterpart, less commonly used than the newton: the kilogram-force (kgf) (sometimes kilopond), is the force exerted by standard gravity on one kilogram of mass. The kilogram-force leads to an alternate, but rarely used unit of mass: the metric slug (sometimes mug or hyl) is that mass that accelerates at 1 m·s−2 when subjected to a force of 1 kgf. The kilogram-force is not a part of the modern SI system, and is generally deprecated; however it still sees use for some purposes as expressing aircraft weight, jet thrust, bicycle spoke tension, torque wrench settings and engine output torque. Other arcane units of force include the sthène, which is equivalent to 1000 N, and the kip, which is equivalent to 1000 lbf.'}])

```python
with open(os.path.join(squad_dir, 'dev.json'), 'w') as f:
    json.dump(new_squad, f)
```

## Establish our QA Model

For our QA model we will setup a simple question-answering pipeline using HuggingFace transformers and a pretrained BERT model. We will be testing it on our SQuAD data so let's load that first.

```python
import json

with open('data/squad/dev.json', 'r') as f:
    squad = json.load(f)
```

```python
from transformers import BertTokenizer, BertForQuestionAnswering
proxies={'http': 'http://127.0.0.1:7890', 'https': 'http://127.0.0.1:7890'}
modelname = 'deepset/bert-base-cased-squad2'
# tokenizer = BertTokenizer.from_pretrained(modelname,proxies=proxies)
# model = BertForQuestionAnswering.from_pretrained(modelname,proxies=proxies)
tokenizer = BertTokenizer.from_pretrained(modelname)
model = BertForQuestionAnswering.from_pretrained(modelname)
```

Transformers comes with a useful class called pipeline which allows us to setup easy to use pipelines for common architectures.

One of those pipelines is the question-answering pipeline which allows us to feed a dictionary containing a `question` and `context` and return an answer. Which we initialize like so:

```python
from transformers import pipeline

qa = pipeline('question-answering', model=model, tokenizer=tokenizer)
```

```python
squad[:2]
```

    [{'question': 'In what country is Normandy located?',
      'answer': 'France',
      'context': 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.'},
     {'question': 'When were the Normans in Normandy?',
      'answer': '10th and 11th centuries',
      'context': 'The Normans (Norman: Nourmands; French: Normands; Latin: Normanni) were the people who in the 10th and 11th centuries gave their name to Normandy, a region in France. They were descended from Norse ("Norman" comes from "Norseman") raiders and pirates from Denmark, Iceland and Norway who, under their leader Rollo, agreed to swear fealty to King Charles III of West Francia. Through generations of assimilation and mixing with the native Frankish and Roman-Gaulish populations, their descendants would gradually merge with the Carolingian-based cultures of West Francia. The distinct cultural and ethnic identity of the Normans emerged initially in the first half of the 10th century, and it continued to evolve over the succeeding centuries.'}]

```python
# we will intialize a list for answers
answers = []

for pair in squad[:5]:
    # pass in our question and context to return an answer
    ans = qa({
        'question': pair['question'],
        'context': pair['context']
    })
    # append predicted answer and real to answers list
    answers.append({
        'predicted': ans['answer'],
        'true': pair['answer']
    })
```

```python
answers
```

    [{'predicted': 'France.', 'true': 'France'},
     {'predicted': '10th and 11th centuries', 'true': '10th and 11th centuries'},
     {'predicted': '10th and 11th centuries',
      'true': 'in the 10th and 11th centuries'},
     {'predicted': 'Denmark, Iceland and Norway',
      'true': 'Denmark, Iceland and Norway'},
     {'predicted': 'Rollo,', 'true': 'Rollo'}]
