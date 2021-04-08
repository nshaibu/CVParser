import random
import os
import spacy
import json
from spacy.training import Example

test_string = """
Nafiu Shaibu
Ghana ▪ nafiushaibu1@gmail.com ▪ +233540241385 ▪ https://nshaibu.github.io
Summary
Forward-thinking Software Engineer with background working effectively in dynamic environments. Fluent in python and javascript programming languages used to develop the PanaBIOS application suite. Proud team player focused on achieving project objectives with speed and accuracy.
Work experience
Turntabl ▪ Accra ▪ Ghana Software engineer
18/01/2021 - present
mPedigree Network Limited Accra ▪ Accra ▪ Ghana Software engineer
09/2018 – 18/01/2021
● Build APIs and setup access gateways.
● Researched, designed and implemented scalable applications for
information identification, extraction, analysis, retrieval and indexing.
● Installed and configured software applications and tested solutions for
effectiveness.
● Conducted regression testing, analyzed results and submitted
observations to the development team.
● Worked with project managers, developers, quality assurance and
customers to resolve technical issues.
● Debugged and troubleshoots PanaBIOS and TrustedTesting for
clients, solving technical issues quickly and accurately.
University IT Services (UITS), KNUST ▪ Kumasi ▪ Ghana Assistant IT Manager
06/2016 – 09/2016
● I was given the responsibility of managing other interns. My daily activities include: Instructing and coordinating the various tasks to be done, assigning tasks to each intern and monitoring their progress.
● Configured and setup networks, computer laboratory.
● Fixed faulty desktop computers, laptops and switches.
● Submitted daily reports to senior management to aid in business
decision-making and planning.
  
     Vodafone Ghana ▪ Accra ▪ Ghana Intern
06/2014 – 09/2014
● I also worked at Vodafone IT department for a week, where I was introduced to the GSM networks and the various IT technologies they use.
● I also worked at the Vodafone datacenter at Gbawe, Accra.
● Delivered expert clerical support by efficiently handling a wide
range of routine and special requirements.
Education
Kwame Nkrumah University Of Science And Technology ▪ Kumasi ▪ Ghana
Computer science
09/2014 – 09/2018
Skills Languages
English
Operating System
Windows 8/10, Kali/Ubuntu Linux, Cisco IOS
DevOps
Git, Docker, Heroku
Programming languages & frameworks
Python [Django, Flask, Machine learning], Javascript [NodeJs, VueJs], Typescript [Angular], C/C++, Bash/Perl/Awk scripting, Tcl/Tk.
Database Systems
MySQL, SQLite, MongoDB, PostgreSQL
Achievements
● PanaBIOS​ is a secure and standard bio-surveillance application suite for disease contagion monitoring, spatial risk factors analytics, mass testing, process traceability & outcomes tracking.
● TrustedTesting​ is a laboratory integrity and test results management system.
    
● OnlineTutor​ is an interactive social learning platforming which incorporates computer vision & AI for better personalized experience for students.
● Internal Communication System​: Designed and developed a concurrent server and client as an internal organizational communication system. The aim of this project was to develop an application that unifies and secure all the communication activities (including chatting, sending and receiving memos or reports securely etc.) within an organization.
● Shortcut Virus Remover​ was a project I worked on during the wake of a virus infestation in our college computer laboratory. The virus infestation was so bad that it made some aspects of our academic work difficult as files couldn't be shared between devices without fear of getting one's machine infected.
At that time, most commercial antivirus programs could not detect and remove that virus. So, I wrote a program in python and the bash scripting language called shortcutVirusRemover, which detects and removes the virus from USB devices. It also protects Microsoft windows computers from getting infected by the Virus. The program is able to detect and remove above 20 other USB viruses. It is designed to also recover user files after the virus has been removed.
Certificates
● Transformative Leadership Training ▪ Kumasi ▪ Ghana International Leadership Foundation (ILF) ▪
https://www.transformingleadership.com/
● Cabling & Fiber Optic Training, KNUST ▪ Kumasi ▪ Ghana
Interests
● Unix & Linux Kernel design
● Machine Learning & AI
Affiliations
● Linux Kernel Genitor: we go through the Linux kernel sources, doing code reviews, fixing up unmaintained code and doing other cleanups and API conversion
"""


def main():
    TRAIN_DATA = json.loads(open("train.json", "r").read())
    
    if not os.path.exists("model/meta.json"):
        nlp = spacy.load('en_core_web_sm')
    else:
        nlp = spacy.load("./model")

    ner = nlp.get_pipe('ner')

    # add labels
    for _, annotations in TRAIN_DATA:
        for ent in annotations.get('entities'):
            ner.add_label(ent[2])

    # get names of other pipes to disable them during training
    other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'ner']
    with nlp.disable_pipes(*other_pipes):
        optimizer = nlp.create_optimizer()
        for itn in range(130):
            print("Starting iteration " + str(itn))
            random.shuffle(TRAIN_DATA)
            losses = {}
            for text, annotations in TRAIN_DATA:
                doc = nlp.make_doc(text)
                try:
                    example = Example.from_dict(doc, annotations)
                except Exception:
                    continue

                nlp.update([example], drop=0.2, sgd=optimizer, losses=losses)
            print(losses)

    nlp.to_disk("model")

    doc = nlp(test_string)

    for ent in doc.ents:
        print(ent.label_, " -- ", ent.text)


if __name__ == '__main__':
    main()
