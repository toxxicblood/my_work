import os
import random
import re
import sys

DAMPING = 0.85
SAMPLES = 10000


def main():
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
    corpus = crawl(sys.argv[1])
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:
            contents = f.read()
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)
            pages[filename] = set(links) - {filename}

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(link for link in pages[filename] if link in pages)

    return pages


def transition_model(corpus, page, damping_factor):
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    probability_distribution = {}
    linked_pages = corpus[page]
    if not linked_pages:
        for p in corpus:
            probability_distribution[p] = round(1 / len(corpus), 4)
        return probability_distribution

    for p in corpus:
        if p in linked_pages:
            probability_distribution[p] = round(
                damping_factor / len(linked_pages) + (1 - damping_factor) / len(corpus),
                4,
            )
        else:
            probability_distribution[p] = round((1 - damping_factor) / len(corpus), 4)

    return probability_distribution


def sample_pagerank(corpus, damping_factor, n):
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    # Choose a random page as the starting point
    """
    current_page = random.choice(list(corpus.keys()))
    page_rank = {page: 0 for page in corpus}
    for i in range(n):
        page_rank[current_page] += 1
        transition = transition_model(corpus, current_page, damping_factor)
        current_page = random.choices(list(transition.keys()), weights=transition.values())[0]
    for page in page_rank:
        page_rank[page] /= n
    return page_rank
    """

    pagecount = {}
    for p in corpus:
        pagecount[p] = 0

    page = random.choice(list(corpus.keys()))
    for i in range(n):
        pagecount[page] += 1
        probability_distribution = transition_model(corpus, page, damping_factor)
        keys = list(probability_distribution.keys())
        values = list(probability_distribution.values())
        page = random.choices(keys, weights=values)[0]
    for p in pagecount:
        pagecount[p] /= n
    return pagecount


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    num_pages = len(corpus)
    # Initialize PageRank values for each page
    page_rank = {page: 1 / num_pages for page in corpus}

    flag = True  # Track convergence
    threshold = 0.001  # Convergence threshold

    while flag:
        flag = False
        new_page_rank = {}

        for page in corpus:
            total = 0
            for possible_page in corpus:
                # Handle pages with no links by treating them as linking to all pages
                links = corpus[possible_page] or corpus.keys()

                if page in links:
                    total += page_rank[possible_page] / len(links)

            # Apply the PageRank formula
            new_rank = ((1 - damping_factor) / num_pages) + (damping_factor * total)
            new_page_rank[page] = new_rank

            # Check for convergence
            if abs(new_rank - page_rank[page]) > threshold:
                flag = True  # Continue iterating if not yet converged

        page_rank = new_page_rank.copy()

    return page_rank


"""
    pageranks = {page: round(1/len(corpus),4) for page in corpus}
    
    def calculate_pagerank(page,corpus, damping_factor):
        part_1 = ((1 - damping_factor)/len(corpus))
        part_2 = 0
        for i in corpus:
            if page in corpus[i]:
                Numlinks = len(corpus[i])
                pagerank_i = pageranks[i]
                part_2_ = pagerank_i/Numlinks
                part_2 += part_2_
        pagerank = part_1 + damping_factor *part_2
        return pagerank
    
    flag = True
    counter = 0

    while flag:
        old_pageranks = pageranks.copy()
        for page in corpus:
            new_pagerank = calculate_pagerank(page,corpus,damping_factor)
            pageranks[page] = new_pagerank
            if abs(old_pageranks [page] - pageranks[page]) > 0.0001:
                flag = True
                counter += 1
    return pageranks"""


if __name__ == "__main__":
    main()
