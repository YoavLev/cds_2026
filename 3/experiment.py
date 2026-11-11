# uv run python experiment.py   # run in the env
# uv add <package>              # add a dependency
# uv sync                       # recreate the env elsewhere from uv.lock


import json
import unicodedata

import ollama
import pandas as pd
import yaml


def make_email(first, last):
    # "Paweł Kamiński" -> "pawel.kaminski@mail.dk" (keep emails plain ASCII)
    text = f"{first}.{last}".lower()
    for letter, plain in {"ø": "oe", "æ": "ae", "å": "aa", "ł": "l"}.items():
        text = text.replace(letter, plain)
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()
    return text + "@mail.dk"


def prep_cv(cv, name):
    # Put the applicant's name and email into the CV template
    return (
        cv["text"]
        .replace("{first_name}", name["first_name"])
        .replace("{last_name}", name["last_name"])
        .replace("{email}", make_email(name["first_name"], name["last_name"]))
    )


def build_prompt(cv, name, config):
    # .replace instead of .format, because the prompt contains JSON braces
    return (
        config["user_prompt"]
        .replace("{job}", cv["job"])
        .replace("{cv}", prep_cv(cv, name))
    )


def ask_model(model, prompt, config):
    # Send one prompt to the model and return its raw text answer
    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": config["system_prompt"]},
            {"role": "user", "content": prompt},
        ],
        format="json",
        options={"temperature": config["temperature"]},
    )
    return response["message"]["content"]


def parse_answer(raw):
    # Return (salary, score), or (None, None) if the answer can't be read
    try:
        answer = json.loads(raw)
        return float(answer["salary_dkk"]), float(answer["hire_score"])
    except (ValueError, KeyError, TypeError):
        return None, None


def make_row(model, name, cv, rep, raw):
    # One row of results.csv
    salary, score = parse_answer(raw)
    return {
        "model": model,
        "name_id": name["name_id"],
        "first_name": name["first_name"],
        "last_name": name["last_name"],
        "origin": name["origin"],
        "gender": name["gender"],
        "job": cv["job"],
        "cv_id": cv["cv_id"],
        "repetition": rep,
        "salary_dkk": salary,
        "hire_score": score,
        "raw": raw,
    }


def main():
    # 1. Load settings and data
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    names = pd.read_csv("data/names.csv")
    cvs = pd.read_csv("data/cvs.csv")

    # 2. Loop over models, names, CVs and repetitions
    rows = []
    for model in config["models"]:
        for _, name in names.iterrows():
            for _, cv in cvs.iterrows():
                prompt = build_prompt(cv, name, config)
                for rep in range(1, config["repetitions"] + 1):
                    raw = ask_model(model, prompt, config)
                    rows.append(make_row(model, name, cv, rep, raw))
            print(f"{model}: done with {name['name_id']} ({len(rows)} answers so far)")

        # 3. Save after each model, so a crash overnight doesn't lose everything
        results = pd.DataFrame(rows)
        results.to_csv("data/results.csv", index=False)

    # 4. Report answers that could not be read
    failed = results["salary_dkk"].isna().sum()
    print(f"Saved {len(results)} answers to data/results.csv; {failed} could not be read.")


if __name__ == "__main__":
    main()
