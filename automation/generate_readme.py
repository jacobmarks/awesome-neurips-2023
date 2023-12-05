import json
from typing import List
import pandas as pd
from pandas.core.series import Series

INTRO_FILE = "intro.md"
COOL_PROJECTS_FILE = "data/cool_projects.csv"
ALL_PAPERS_FILE = "data/neurips2023data.json"

####### COOL PROJECTS #######


TABLE_HEADER = [
    "| **Title** | **Paper** | **Code** | **Project Page** | **Hugging Face** |",
    "|:---------:|:---------:|:--------:|:----------------:|:----------------:|",
]

TITLE_COLUMN_NAME = "title"
PAPER_COLUMN_NAME = "arxiv"
CODE_COLUMN_NAME = "github"
PROJECT_COLUMN_NAME = "project_page"
HF_COLUMN_NAME = "hugging_face"

GITHUB_CODE_PREFIX = "https://github.com/"
GITHUB_BADGE_PATTERN = (
    "[![GitHub](https://img.shields.io/github/stars/{}?style=social)]({})"
)
ARXIV_BADGE_PATTERN = "[![arXiv](https://img.shields.io/badge/arXiv-{}-b31b1b.svg)](https://arxiv.org/abs/{})"
HF_BADGE_PATTERN = "[![Hugging Face](https://img.shields.io/badge/🤗-Open%20In%20Spaces-blue.svg)]({})"
PROJECT_PATTERN = "[Project]({})"


def _add_row_to_table(data_file, entry: Series) -> str:
    """Formats entry into markdown table row."""
    title = entry.loc[TITLE_COLUMN_NAME]
    arxiv_ref = str(entry.loc[PAPER_COLUMN_NAME])
    code_url = entry.loc[CODE_COLUMN_NAME]
    project_url = entry.loc[PROJECT_COLUMN_NAME]
    hf_url = entry.loc[HF_COLUMN_NAME]

    hf_badge = HF_BADGE_PATTERN.format(hf_url) if type(hf_url) == str else ""
    code_badge = (
        GITHUB_BADGE_PATTERN.format(code_url, GITHUB_CODE_PREFIX + code_url)
        if type(code_url) == str
        else ""
    )
    paper_badge = ARXIV_BADGE_PATTERN.format(arxiv_ref, arxiv_ref)
    project = (
        PROJECT_PATTERN.format(project_url) if type(project_url) == str else ""
    )

    return (
        f"| {title} | {paper_badge} | {code_badge}| {project} | {hf_badge} |"
    )


def load_table_entries(path: str) -> List[str]:
    """Loads table entries from csv file."""
    df = pd.read_csv(path, quotechar='"', dtype=str)
    sorted_df = df.sort_values(by=TITLE_COLUMN_NAME)
    sorted_df.columns = sorted_df.columns.str.strip()
    return [_add_row_to_table(path, row) for _, row in sorted_df.iterrows()]


def add_cool_projects(markdown_content):
    markdown_content += "\n\n## Cool NeurIPS Projects\n\n"
    markdown_content += "\n".join(TABLE_HEADER) + "\n"
    markdown_content += "\n".join(load_table_entries(COOL_PROJECTS_FILE))
    return markdown_content


####### CONFERENCE SCHEDULE #######

SESSION_DICT = {
    1: "Poster Session 1: Tuesday, Dec 12, 08:45 CT",
    2: "Poster Session 2: Tuesday, Dec 12, 15:15 CT",
    3: "Poster Session 3: Wednesday, Dec 13, 08:45 CT",
    4: "Poster Session 4: Wednesday, Dec 13, 15:00 CT",
    5: "Poster Session 5: Thursday, Dec 14, 08:45 CT",
    6: "Poster Session 6: Thursday, Dec 14, 15:00 CT",
    -1: "Posters Not Being Presented",
}

SESSION_NUMBERS = [1, 2, 3, 4, 5, 6, -1]


def _get_session_papers(papers_data, session_number):
    session_papers = [
        v for v in papers_data.values() if v["session"] == session_number
    ]
    sorted_papers = sorted(session_papers, key=lambda x: x["title"])
    return sorted_papers


def _get_session(date):
    if "Poster Session" in date:
        return int(date.split("Poster Session ")[1])
    else:
        return -1


# Function to format the entry into Markdown
def format_entry(entry):
    oral_str = ""
    authors = ", ".join(entry["authors"])
    abstract_string = ""
    # abstract = entry["abstract"].replace("\n", "\n\n")
    # if abstract == "":
    #     abstract_string = ""
    # else:
    #     abstract_string = f"**<details><summary>Abstract**</summary>\n\n{abstract}</details>\n\n"
    title_str = entry["title"]
    if entry["spotlight"]:
        title_str = "[Spotlight] " + title_str
    if entry["oral"]:
        title_str = "[Oral] " + title_str
        oral_str = (
            f"**Oral Presentation:** {entry['oral_presentation_time']}\n\n"
        )

    return (
        f"### {title_str}\n\n"
        + f"**Authors:** {authors}\n\n{oral_str}{abstract_string}"
    )


def _add_session(papers_by_session, session_number):
    session_str = f"{SESSION_DICT[session_number]}"
    current_session = f"<details><summary><h3 style='display: inline;'> {session_str}</h3></summary>\n\n"

    # Add each paper to the session
    for entry in papers_by_session[session_number]:
        current_session += format_entry(entry)

    # Close the session details block
    current_session += "</details>\n\n"

    return current_session


def add_conference_schedule(markdown_content):
    markdown_content += "\n\n## Conference Schedule\n\n"
    with open(ALL_PAPERS_FILE, "r") as f:
        papers = json.load(f)

    keys = list(papers.keys())
    for k in keys:
        date = papers[k]["presentation_time"]
        session = _get_session(date)
        papers[k]["session"] = session

    papers_by_session = {
        n: _get_session_papers(papers, n) for n in SESSION_NUMBERS
    }

    for session_number in SESSION_NUMBERS:
        markdown_content += _add_session(papers_by_session, session_number)

    return markdown_content


#####

with open(INTRO_FILE, "r") as f:
    markdown_content = f.read()

# markdown_content = HEADER + intro_content
markdown_content = add_cool_projects(markdown_content)
markdown_content = add_conference_schedule(markdown_content)

# Write the formatted content into a README.md file
with open("README.md", "w") as md_file:
    md_file.write(markdown_content)

print("README.md file has been created successfully.")
