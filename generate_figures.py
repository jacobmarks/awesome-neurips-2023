import json
import numpy as np
from collections import Counter
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rc_file('matplotlibrc')
plt.rcParams['text.usetex'] = True

from wordcloud import WordCloud

with open("data/neurips2022data.json", "r") as f:
    neurips_2022_data = json.load(f)

with open("data/neurips2023data.json", "r") as f:
    neurips_2023_data = json.load(f)

with open("data/stopwords.txt", "r") as f:
    stopwords = f.read().split("\n")

titles_2022 = [d.lower() for d in neurips_2022_data.keys()]
titles_2023 = [d.lower() for d in neurips_2023_data.keys()]

def generate_wordcloud(titles, stopwords, filename):
    title_text = ' '.join(titles)
    title_text = ' '.join([word for word in title_text.split() if word not in stopwords])

    wordcloud = WordCloud(
        max_font_size=100, 
        min_font_size = 10, 
        max_words=100, 
        background_color="white", 
        width = 533*2, 
        height = 253*2, 
        mode="RGBA"
        ).generate(title_text)
    wordcloud.to_file(filename)

for year, titles in zip(["2022", "2023"], [titles_2022, titles_2023]):
    generate_wordcloud(titles, stopwords, f"images/wordcloud_{year}.png");


keys = list(neurips_2023_data.keys())

abstracts = [neurips_2023_data[key]["abstract"] for key in keys]
abstracts = [abstract.lower() for abstract in abstracts if abstract!=""]
abstract_lengths = [len(abstract.split()) for abstract in abstracts]

plt.figure(figsize=(10, 6))
plt.hist(abstract_lengths, bins=100, density=True);
plt.xlabel("Abstract Length (words)");
plt.ylabel("Density");
plt.title("Abstract Lengths for NeurIPS 2023 Papers");
plt.savefig("images/abstract_histogram_2023.png");
plt.cla()
plt.close()

num_authors_2022 = [
    len(v) for v in neurips_2022_data.values()
]

num_authors_2023 = [
    len(neurips_2023_data[k]['authors']) for k in keys
]



author_counts_2022 = Counter(num_authors_2022)
author_counts_2023 = Counter(num_authors_2023)

all_keys = sorted(set(author_counts_2022.keys()).union(set(author_counts_2023.keys())))
values_2022 = [author_counts_2022.get(k, 0) for k in all_keys]
values_2023 = [author_counts_2023.get(k, 0) for k in all_keys]

# Set up the bar width and positions
bar_width = 0.35
r1 = range(len(all_keys))
r2 = [x + bar_width for x in r1]

# Create bar chart
plt.figure(figsize=(15, 6))
plt.bar(r1, values_2022, color='blue', width=bar_width, edgecolor='grey', label='2022')
plt.bar(r2, values_2023, color='red', width=bar_width, edgecolor='grey', label='2023')
# Add labels and title
plt.xlabel('Number of Authors')
plt.ylabel('Number of Papers')
plt.title('Comparison of Number of Authors per Paper for 2022 vs 2023')
plt.xticks([r + bar_width / 2 for r in r1], all_keys);
# Add a legend
plt.legend();
plt.savefig("images/num_authors_2022_2023.png")
plt.cla()
plt.close()




title_lengths_2022 = [len(title.split()) for title in titles_2022]
title_lengths2023 = [len(neurips_2023_data[key]["title"].split()) for key in keys]

plt.figure(figsize=(10, 6))
plt.hist(title_lengths_2022, bins=100, density=True, label="2022");
plt.hist(title_lengths2023, bins=100, density=True, label="2023");
plt.xlabel("Title Length (words)");
plt.ylabel("Density");
plt.legend();
plt.savefig("images/title_length_histogram_2022_2023.png");
plt.cla()
plt.close()

historical_data = {}

for year in range(2014, 2023):
    syear = str(int(year))
    with open(f"data/neurips{syear}data.json", "r") as f:
        historical_data[syear] = json.load(f)

historical_data["2023"] = {
    val["title"]: val["authors"]
    for val in neurips_2023_data.values()
}

syears = [str(int(year)) for year in range(2014, 2024)]

title_lengths_by_year = {
    syear: [len(title.split()) for title in historical_data[syear].keys()]
    for syear in syears
}

mean_title_length_by_year = {
    syear: np.mean(title_lengths_by_year[syear])
    for syear in syears
}

plt.figure(figsize=(10, 6))
plt.plot(syears, list(mean_title_length_by_year.values()), "-.")
plt.xlabel("Year");
plt.ylabel("Mean Title Length (words)");
plt.title(f"Mean NeurIPS Paper Title Length {syears[0]}—{syears[-1]}")
plt.savefig("images/title_length_history_2014_2023.png");
plt.cla()
plt.close()

num_authors_by_year = {
    syear: [len(authors) for authors in historical_data[syear].values()]
    for syear in syears
}

mean_num_authors_by_year = {
    syear: np.mean(num_authors_by_year[syear])
    for syear in syears
}

plt.figure(figsize=(10, 6))
plt.plot(syears, list(mean_num_authors_by_year.values()), "-.")
plt.xlabel("Year");
plt.ylabel("Mean Number of Authors");
plt.title(f"Mean Number of Authors per NeurIPS Paper {syears[0]}—{syears[-1]}")
plt.savefig("images/num_authors_history_2014_2023.png");
plt.cla()
plt.close()


num_unique_authors_by_year = {
    syear: len(set([v for vals in historical_data[syear].values() for v in vals]))
    for syear in syears
}

plt.figure(figsize=(10, 6))
plt.plot(syears, list(num_unique_authors_by_year.values()), "-.")
plt.xlabel("Year");
plt.ylabel("Number of Unique Authors");
plt.title(f"Number of Unique Authors of NeurIPS Papers {syears[0]}—{syears[-1]}")
plt.savefig("images/unique_authors_history_2014_2023.png");
plt.cla()
plt.close()