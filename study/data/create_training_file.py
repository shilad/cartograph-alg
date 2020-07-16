import pandas as pd


def get_concatenated(directory, project, clustering_alg):
    categories = pd.read_csv(directory + project + "/" + clustering_alg + '/labels/h_cat/top_labels.csv')
    key_phrases = pd.read_csv(directory + project + "/" + clustering_alg + '/labels/key_phrases/top_labels.csv')
    key_words = pd.read_csv(directory + project + "/" + clustering_alg +  '/labels/key_words/top_labels.csv')
    links = pd.read_csv(directory + project + "/" + clustering_alg + '/labels/links/top_labels.csv')
    if (project!= "media"):
        if(clustering != "LDA"):
            lda = pd.read_csv(directory + project + "/" + clustering_alg + '/labels/lda/top_labels.csv')
        else:
            lda = pd.read_csv(directory + project + "/" + clustering_alg + '/labels/LDA_labels/top_labels.csv')

        concatenated = pd.concat([categories, key_phrases, key_words, links, lda])
    else:
        concatenated = pd.concat([categories, key_phrases, key_words, links])

    concatenated = concatenated.fillna(0)
    return concatenated

def main(directory, clustering):
    all_projects = []
    for j in ['kmeans_plain', 'kmeans_augmented', 'LDA']:
        for i in [ "media", "food", "internet", "technology"]:
            contate = get_concatenated(directory, i, j)
            all_projects.append(contate)
    allprojects = pd.concat(all_projects)
    hit_labels = pd.read_csv("/Users/luli/PycharmProjects/cartograph-alg/study/hit_labels.csv")
    hit_labels = hit_labels[["project", "cluster_alg", "cluster_num", "label_name", "avg_borda"]]
    allprojects['h_cat_tfidf'] = allprojects['h_cat'] * allprojects['tfidf']
    allprojects['key_words_tfidf'] = allprojects['key_words'] * allprojects['tfidf']
    allprojects['key_phrases_tfidf'] = allprojects['key_phrases'] * allprojects['tfidf']
    allprojects['links_tfidf'] = allprojects['links'] * allprojects['tfidf']
    allprojects['lda_tfidf'] = allprojects['lda'] * allprojects['tfidf']

    allprojects = allprojects.drop(columns=["Unnamed: 0"])
    allprojects = pd.DataFrame(allprojects)
    final = pd.merge(hit_labels, allprojects, how="inner", left_on=["cluster_alg", "cluster_num", "label_name", "project"], right_on=["cluster_alg", "country", "label_name", "project"])
    final = final.groupby(['project', 'cluster_alg', 'cluster_num', 'label_name']).sum().reset_index()
    final = final.drop(columns=['tfidf', 'links', 'h_cat', 'key_words', 'key_phrases', 'lda', 'country','LDA_labels'])
    final.to_csv("study/data/final_labels.csv")
if __name__ == '__main__':
    import sys

    if len(sys.argv) != 3:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    directory, clustering = sys.argv[1:]
    main(directory, clustering)
