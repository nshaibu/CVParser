import io
import re
import math
import itertools
import functools
import pandas as pd
import docx2txt
import nltk
from spacy.matcher import PhraseMatcher
from pdfminer.converter import TextConverter
from pdfminer.pdfinterp import PDFPageInterpreter
from pdfminer.pdfinterp import PDFResourceManager
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from .resources import RESOURCES


STOPWORDS = set(stopwords.words('english'))


L2Norm = lambda x, x1, y, y1: math.sqrt(pow((x - x1), 2) + pow((y - y1), 2))

MONTH = r'(' + RESOURCES['patterns']['MONTHS_SHORT'] + r'|' + RESOURCES['patterns']['MONTHS_LONG']\
        + r'|0[1-9]|1[0-2]' + r')'

DATE_SEP = r"[-/\s\S]{0,2}"

DATE_PATTERN = r'(' + MONTH + DATE_SEP + RESOURCES['patterns']['YEAR'] \
               + r'|' + RESOURCES['patterns']['YEAR'] + DATE_SEP + MONTH\
               + r'|' + RESOURCES['patterns']['DAY'] + DATE_SEP + MONTH + DATE_SEP + RESOURCES['patterns']['YEAR'] \
               + r'|' + MONTH + RESOURCES['patterns']['DAY'] + RESOURCES['patterns']['YEAR']\
               + r'|' + RESOURCES["patterns"]['YEAR'] + r')'


DATE_RANGE_PATTERN = r"(" + DATE_PATTERN + r"[-\s\S]{0,8}" + DATE_PATTERN + r"|" + DATE_PATTERN + r")"


def remove_special_chars(val: str):
    return val.replace("\\", " ") \
        .replace("(", " ").replace(")", " ").replace("{", " ").replace("}", " ") \
        .replace("^", "").replace("?", "").replace("-", " ").replace("[", " ") \
        .replace("]", " ").replace("&", "").replace('"', "").replace("'", "")


@functools.lru_cache()
def make_case_insensitive(val: str,  disable_comp: bool=True) -> list:
    vals = list()
    vals.append(val)
    vals.append(str(val).upper())
    vals.append(str(val).lower())
    vals.append(str(val).title())

    lt = val.split(" ")

    operations = ["LOWER", "TITLE"]

    size = len(lt)
    if not disable_comp and size > 1:
        operations *= size
        operations = list(itertools.combinations(operations, size))
        for ops in operations:
            values = []
            for index, op in enumerate(ops):
                _val = lt[index]
                if op == "LOWER":
                    values.append(str(_val).lower())
                else:
                    values.append(str(_val).title())

            vals.append(" ".join(values))
    return vals


def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as fh:
        for page in PDFPage.get_pages(fh, 
                                      caching=True,
                                      check_extractable=True):
            resource_manager = PDFResourceManager()
            fake_file_handle = io.StringIO()
            converter = TextConverter(resource_manager, fake_file_handle, laparams=LAParams())
            page_interpreter = PDFPageInterpreter(resource_manager, converter)
            page_interpreter.process_page(page)
 
            text = fake_file_handle.getvalue()
            yield text
 
            # close open handles
            converter.close()
            fake_file_handle.close()


def extract_text_from_doc(doc_path):
    temp = docx2txt.process(doc_path)
    text = [line.replace('\t', ' ') for line in temp.split('\n') if line]
    return ' '.join(text)


def extract_mobile_number(text):
    # phone_nums = re.findall(re.compile(r'(?:(?:\+?([1-9]|[0-9][0-9]|[0-9][0-9][0-9])\s*(?:[.-]\s*)?)?(?:\(\s*([2-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9])\s*\)|([0-9][1-9]|[0-9]1[02-9]|[2-9][02-8]1|[2-9][02-8][02-9]))\s*(?:[.-]\s*)?)?([2-9]1[02-9]|[2-9][02-9]1|[2-9][02-9]{2})\s*(?:[.-]\s*)?([0-9]{4})(?:\s*(?:#|x\.?|ext\.?|extension)\s*(\d+))?'), text)
    mob_num_regex = r'''(\d{3}[-\.\s]??\d{3}[-\.\s]??\d{4}|\(\d{3}\)
                        [-\.\s]*\d{3}[-\.\s]??\d{4}|\d{3}[-\.\s]??\d{4})'''
    phone_nums = re.findall(re.compile(mob_num_regex), text)
    if phone_nums:
        return ["".join(num) for num in phone_nums]
    return phone_nums


def extract_email(text):
    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", text)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None


def extract_skills(parser, span, noun_chunks):
    skill_set = set()
    matcher = PhraseMatcher(parser.nlp.vocab)

    with open(RESOURCES['skills_file'], "r") as fd:
        skills = fd.readlines()
        skills = [remove_special_chars(str(skill).strip().lower()) for skill in skills if skill]

        patterns = []
        for skill in skills:
            pattern = [parser.nlp.make_doc(token) for token in make_case_insensitive(skill)]
            patterns.extend(pattern)

        matcher.add("MATCH_SKILLS", patterns)
        matches = matcher(span)
        for match_id, start, end in matches:
            sub_span = parser.doc[start:end]
            if sub_span:
                skill_set.add(sub_span.text)

    return [i for i in skill_set]


def detect_resume_sections(parser) -> []:
    from .parser import ResumeSection
    resume_sections = RESOURCES['resume_sections']
    sections = []
    for section_name in resume_sections:
        matcher = PhraseMatcher(parser.nlp.vocab)
        patterns = []
        for text in resume_sections[section_name]:
            pattern = [parser.nlp.make_doc(token) for token in make_case_insensitive(text)]
            patterns.extend(pattern)

        matcher.add("resume-section-%s" % section_name, patterns)
        matches = matcher(parser.doc)
        for match_id, start, end in matches:
            span = parser.doc[start:end]
            if span:
                s = ResumeSection(section_name=section_name, start_index=start, end_index=end)
                sections.append(s)
                break

    return sections


def _extract_education_date(parser, doc, start, end):
    left_entries = []
    right_entries = []

    left_min_date = right_min_date = None

    # Search Left
    left_span = doc[:start]
    left_doc = parser.nlp(left_span.text.strip().replace("\n", " "))

    for ent in left_doc.to_json()['ents']:
        if ent['label'] == 'DATE':
            left_entries.append(ent)

    # search right
    right_span = doc[end:]
    right_doc = parser.nlp(right_span.text.strip().replace("\n", ""))

    for ent in right_doc.to_json()['ents']:
        if ent['label'] == "DATE":
            right_entries.append(ent)

    try:
        left_min_date = min(left_entries, key=lambda x: L2Norm(start, x['start'], end, x['end']))
    except Exception:
        pass

    try:
        right_min_date = min(right_entries, key=lambda x: L2Norm(start, x['start'], end, x['end']))
    except Exception:
        pass

    if left_min_date is None and right_min_date is None:
        return None
    elif left_min_date is None:
        return right_min_date
    elif right_min_date is None:
        return left_min_date
    return min([right_min_date, left_min_date], key=lambda x: L2Norm(start, x['start'], end, x['end']))


def _extract_course(parser, doc, end_index):
    course_text = ""
    matcher = PhraseMatcher(parser.nlp.vocab)
    doc = parser.nlp(str(doc.text).strip().replace("\n", " "))

    with open(RESOURCES["courses_file"], "r") as fd:
        courses = fd.readlines()
        courses = [str(course).strip() for course in courses if course]

    patterns = []
    for course in courses:
        pattern = [parser.nlp.make_doc(token) for token in make_case_insensitive(course, disable_comp=False)]
        patterns.extend(pattern)

    matcher.add("MATCH_COURSES", patterns)
    doc_span = parser.nlp(doc[end_index:].text)
    matches = matcher(doc_span)

    for match_id, start, end in matches:
        span = doc_span[start:end]
        if span:
            course_text = span.text
            break
    return course_text


def extract_education(parser, text):
    schools_set = []
    matcher = PhraseMatcher(parser.nlp.vocab)
    doc = parser.nlp(str(text).strip())

    with open(RESOURCES["schools_file"], "r") as fd:
        schools = fd.readlines()
        schools = [str(school).strip() for school in schools if school]

        patterns = []
        for school in schools:
            pattern = [parser.nlp.make_doc(token) for token in make_case_insensitive(school)]
            patterns.extend(pattern)

        matcher.add("MATCH_SCHOOLS", patterns)
        matches = matcher(doc)
        for match_id, start, end in matches:
            span = doc[start:end]
            if span:
                school_dict = dict()
                school_dict['name'] = span.text
                school_dict['course'] = _extract_course(parser, doc, end)
                date_entity_cord = _extract_education_date(parser, doc, start, end)
                if date_entity_cord:
                    sub_span = doc[date_entity_cord['start']:date_entity_cord['end']]
                    if sub_span:
                        school_dict['date'] = sub_span.text

                if "date" not in school_dict:
                    left_span = doc[:start]

                    date = re.search(re.compile(DATE_RANGE_PATTERN), left_span.text.strip())
                    if date:
                        school_dict['date'] = date.group(0)
                    else:
                        right_span = doc[end:]
                        date = re.search(re.compile(DATE_RANGE_PATTERN), right_span.text.strip())
                        if date:
                            school_dict['date'] = date.group(0)
                schools_set.append(school_dict)

    return schools_set


def extract_experience_sentences(resume_text):
    print(resume_text)
    wordnet_lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    # word tokenization
    word_tokens = nltk.word_tokenize(resume_text)

    # remove stop words and lemmatize
    filtered_sentence = [
            w for w in word_tokens if w not
            in stop_words and wordnet_lemmatizer.lemmatize(w)
            not in stop_words
        ]
    sent = nltk.pos_tag(filtered_sentence)

    # parse regex
    cp = nltk.RegexpParser('P: {<NNP>+}')
    cs = cp.parse(sent)

    test = []

    for vp in list(
        cs.subtrees(filter=lambda x: x.label() == 'P')
    ):
        test.append(" ".join([
            i[0] for i in vp.leaves()
            if len(vp.leaves()) >= 2])
        )

    # Search the word 'experience' in the chunk and
    # then print out the text after it
    x = [
        x[x.lower().index('experience') + 10:]
        for i, x in enumerate(test)
        if x and 'experience' in x.lower()
    ]
    return x


def extract_opportunity_available(parser):
    opportunities_set = set()
    matcher = PhraseMatcher(parser.nlp.vocab)

    opportunities = [str(op).strip() for op in RESOURCES["available_opportunities"] if op]

    patterns = []
    for op in opportunities:
        pattern = [parser.nlp.make_doc(token) for token in make_case_insensitive(op)]
        patterns.extend(pattern)

    matcher.add("MATCH_AVAILABLE_OPPORTUNITIES", patterns)
    matches = matcher(parser.doc)
    for match_id, start, end in matches:
        span = parser.doc[start:end]
        if span:
            opportunities_set.add(span.text)

    word_intercepts = opportunities_set & STOPWORDS
    for word in word_intercepts:
        opportunities_set.remove(word)

    return list(opportunities_set)
