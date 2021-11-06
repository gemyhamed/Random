from csv import DictReader, DictWriter
import argparse
import pycountry
import fasttext
import pycld2
from ftfy.fixes import decode_escapes
import ftfy
import pandas as pd
import numpy as np


model = fasttext.load_model('lid.176.bin')


def read_csv_field(filepath, field1,field2):
    with open(filepath, newline='', encoding='utf-8') as csvfile:
        reader = DictReader(csvfile)
        for row in reader:
            yield row[field1],row[field2]


def get_language(text):
    try:
        languages, predictions = model.predict(text, k=1)
        lang_code = languages[0].replace('__label__', '')
        lang_name = pycountry.languages.get(alpha_2=lang_code).name
        prediction = predictions[0]
    except:
        try:
            _, _, languages = pycld2.detect(text, bestEffort=True)
            lang_code = languages[0][1]
            lang_name = pycountry.languages.get(alpha_2=lang_code).name
            prediction = ''
        except:
            lang_code = lang_name = prediction = ''

    return lang_code, lang_name, prediction

def removespace(x):
    
    new = x[:-7]+x[-7:].replace(' ','')
    return new
   



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', help='input csv file')
    parser.add_argument('--output_file', help='output folder path', default='results.csv')
    args = parser.parse_args()

    with open(args.output_file, 'w', newline='', encoding='utf-8') as results_csv:
        writer = DictWriter(results_csv, fieldnames=['title', 'language_code', 'language_name', 'prediction','corrected_title','count'])
        writer.writeheader()
        for title,count in read_csv_field(args.input_file, 'title','count'):
            lang_code, lang_name, prediction = get_language(title)
            corrected = removespace(ftfy.fix_text(decode_escapes(title)))
        
            writer.writerow({'title': title,
                             'language_code': lang_code,
                             'language_name': lang_name,
                             'prediction': prediction,
                             'corrected_title':corrected,
                             'count':count})
        
