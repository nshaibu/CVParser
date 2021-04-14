import os, json, re
import nltk
import ssl
import mimetypes
import spacy
from spacy.matcher import Matcher
from collections import OrderedDict
from typing import NamedTuple

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

from cvparser.utils import (extract_text_from_pdf, extract_text_from_doc, extract_text_from_files, extract_mobile_number,
                            extract_email, extract_skills, detect_resume_sections, extract_education,
                            extract_experience_sentences, extract_opportunity_available, extract_entities_wih_custom_model)

from .resources import RESOURCES

check = lambda key, container: key in container and container[key]


class FileMimeTypeError(Exception):
    pass


class ResumeSection(NamedTuple):
    section_name: str
    start_index: int
    end_index: int

    @staticmethod
    def get_resume_sections(section_name: str, collection: list):
        sorted_lt = sorted(collection, key=lambda x: x.end_index)
        index = -1
        value = None
        for _index, _value in enumerate(sorted_lt):
            if str(_value.section_name).lower() == str(section_name).lower():
                index = _index
                value = _value
                break
        return index, value

    @staticmethod
    def next_section(previous_index: int, collection: list):
        sorted_lt = sorted(collection, key=lambda x: x.end_index)
        index = previous_index + 1
        if index < len(collection):
            return index, sorted_lt[index]
        return -1, None

    @classmethod
    def get_end_index(cls, index: int, collection: list):
        _, value = cls.next_section(index, collection)
        return value.start_index if value else None


class CVParser:
    SUPPORTED_MIMETYPES = [
        "application/msword",
        "application/pdf",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "image/jpeg",
        "image/png",
    ]

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.file_content = self._process_file()

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.custom_model_path = os.path.join(BASE_DIR, "train/model")

        self.nlp = spacy.load("en_core_web_sm")
        self.custom_nlp = spacy.load(self.custom_model_path)

        self.custom_doc = self.custom_nlp(self.file_content)
        self.custom_matcher = Matcher(self.custom_nlp.vocab)

        self.matcher = Matcher(self.nlp.vocab)
        self.doc = self.nlp(self.file_content)
        self.noun_chunks = list(self.doc.noun_chunks)
        self.sections = ['name', 'email', 'mobile_numbers', 'skills', 'education',
                         'experience', 'opportunities']
        self.data = OrderedDict()

    def _process_file(self) -> str:
        if not os.path.exists(self.file_path):
            raise FileNotFoundError("File does not exist.")

        file_mime, _ = mimetypes.guess_type(self.file_path)
        file_mime = str(file_mime).lower()

        if file_mime not in self.SUPPORTED_MIMETYPES:
            raise FileMimeTypeError("Sorry file type not supported.")

        try:
            if file_mime == "application/pdf":
                text = " ".join([page for page in extract_text_from_pdf(self.file_path)])
            elif file_mime in ["application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                               "application/msword"]:
                text = extract_text_from_doc(self.file_path)
            else:
                text = extract_text_from_files(self.file_path)
        except Exception:
            text = extract_text_from_files(self.file_path)
        return text

    @staticmethod
    def download_nlk_data():
        nltk.download('maxent_ne_chunker')
        nltk.download('words')
        nltk.download('stopwords')
        nltk.download('punkt')
        nltk.download('wordnet')
        nltk.download('averaged_perceptron_tagger')

    def parse(self):
        check_key = lambda key: check(key, self.data)

        detected_sections = detect_resume_sections(self)
        custom_entities = extract_entities_wih_custom_model(self.custom_doc)

        for section in self.sections:
            if not check_key(section):
                index, sect = ResumeSection.get_resume_sections(section, detected_sections)

                if section == "name":
                    try:
                        self.data['name'] = custom_entities['Name'][0]
                    except (IndexError, KeyError):
                        self.matcher.add("USER_NAME", [RESOURCES['patterns']['NAME_PATTERN']])
                        matches = self.matcher(self.doc)
                        for _, start, end in matches:
                            span = self.doc[start:end]
                            self.data[section] = span.text
                            break
                    try:
                        self.data['designation'] = custom_entities['Designation']
                    except KeyError:
                        pass
                elif section == "mobile_numbers":
                    self.data[section] = extract_mobile_number(self.file_content)
                elif section == "email":
                    self.data[section] = extract_email(self.file_content)

                elif section == "skills":
                    if index >= 0:
                        start_index = sect.end_index
                        end_index = ResumeSection.get_end_index(index, detected_sections)

                        span = self.doc[start_index:end_index] if end_index else self.doc[start_index:]
                        self.data[section] = extract_skills(self, span, list(span.noun_chunks))
                    skill_set = set(section in self.data and self.data[section] or [])
                    try:
                        for skill in custom_entities['Skills']:
                            skill = str(skill).strip().replace("\n", " ")
                            skill_set.add(skill)
                        self.data[section] = list(skill_set)
                    except Exception as e:
                        print(e)
                        pass
                elif section == "education":
                    if index >= 0:
                        start_index = sect.end_index
                        end_index = ResumeSection.get_end_index(index, detected_sections)

                        span = self.doc[start_index:end_index] if end_index else self.doc[start_index:]
                        self.data[section] = extract_education(self, span.text)

                    try:
                        self.data['college_name'] = custom_entities['College Name']
                    except Exception:
                        pass

                    try:
                        self.data['graduation_year'] = custom_entities['Graduation Year']
                    except:
                        pass

                    try:
                        self.data['degree'] = custom_entities['Degree']
                    except KeyError:
                        pass
                elif section == "experience":
                    experience = dict()
                    experience['sentences'] = extract_experience_sentences(" ".join(self.file_content.split()))

                    if index >= 0:
                        pass
                    try:
                        experience['companies_worked_at'] = custom_entities['Companies worked at']
                    except Exception:
                        pass
                    self.data[section] = experience
                elif section == "opportunities":
                    self.data[section] = extract_opportunity_available(self)

        # print(detected_sections)

    def to_dict(self):
        return dict(self.data)

    def json(self):
        return json.dumps(self.to_dict())

