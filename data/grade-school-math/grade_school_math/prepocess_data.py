import json
import re
import os


def read_json(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object and append it to the list
            data.append(json.loads(line))

    for i in range(len(data)):
        yield data[i]


def preprocess_numbers(text: str):
    # replace whitespaces in number like 1 000 000 with 1000000
    number_whitespace_regex = r"(^|\s)(\d{1,3})( \d{3})+(\.\d+)?"
    matches = re.findall(number_whitespace_regex, text)
    for match in matches:
        match = "".join(match[1:])
        text = text.replace(match, match.replace(" ", ""))

    # replace commas in number like 1,000,000 with 1000000
    number_comma_regex = r"\d{1,3}(,\d{3})+(\.\d+)?"
    matches = re.findall(number_comma_regex, text)
    for match in matches:
        text = text.replace(match[0], match[0].replace(",", ""))

    return text


def main():
    # print current dir
    print(os.getcwd())

    file_path = "data/"
    file_name = "test_clean.jsonl"

    for line in read_json(file_path + file_name):
        question = preprocess_numbers(line['question'])
        answer = preprocess_numbers(line['answer'])
        line['question'] = question
        line['answer'] = answer

        with open(file_path + "preprocessed/" + file_name, 'a') as file:
            file.write(json.dumps(line) + "\n")


if __name__ == "__main__":
    main()
