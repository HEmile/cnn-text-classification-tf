import os

class review_location:
    #fileName, Maplocation
    def __init__(self, filename, mapLocation):
        self.fileName = filename
        self.mapLocation = ""
        if mapLocation != "null":
            self.mapLocation = mapLocation

    def getPath(self):
        return self.mapLocation + self.fileName

def getReviewsRecursive(path, after, extension, reviews=[], depth=0):
    for filemap in [doc for doc in os.listdir(path + after)]:
        if os.path.isdir(path+after+filemap+'/'):
            reviews = getReviewsRecursive(path, after+filemap+'/', extension, reviews, depth+1)
        elif filemap.endswith(extension):
            # print("depth" + str(depth) + " = " + filemap)
            reviews.append(review_location(filemap, after))
    return reviews

def cleanData(lines):
    changeables = ['.', ',', '(', ')', '!', '?', ';', ':']
    unmentionables = ["\'", "\""]
    for changeable in changeables:
        lines = lines.replace(changeable, ' ' + changeable + ' ')
    for unmentionable in unmentionables:
        lines = lines.replace(unmentionable, ' ')
    # lines = lines.replace("<\ br>", )

    paragraphs = lines.split("<br />")
    lines = []
    for paragraph in paragraphs:
        line = paragraph.split()
        if len(line) > 3:
            lines.append(line)

    return lines

path = "C:/Users/Roderick/Downloads/aclImdb_v1/aclImdb/actual_data/"
reviews = getReviewsRecursive(path=path, after="", extension=".txt")

with open("data/rt-polaritydata/aclImdb.txt", 'w') as new_file:
    for i in range(len(reviews)):
        with open(path + reviews[i].getPath(), 'r') as review:
            # print(path + reviews[i].getPath())
            try:
                lines = review.readlines()
            except:
                print("skipping", path + reviews[i].getPath())
                continue
            lines = cleanData(lines[0])

            for line in lines:
                new_file.write(' '.join(line) + '\n')
