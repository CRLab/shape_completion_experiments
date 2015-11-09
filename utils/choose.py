import os


# options: [(item_name, option)]
# topic: are we choosing a model? a pipeline?..
def choose(options, topic):

    print
    print "Choose a " + topic
    print

    for i in range(len(options)):
        print str(i) + ": " + options[i][0]

    print
    option_index = int(raw_input("Enter Id of option (ex 0, 1, or 2): "))

    option = options[option_index][1]

    return option


# We often want to choose a model or a dataset
# so generic method to choose a file or folder out of a directory
def choose_from(directory):

    options = os.listdir(directory)

    print
    print "Choose from " + str(directory)
    print

    for i in range(len(options)):
        print str(i) + ": " + options[i]

    print
    option_index = int(raw_input("Enter Id of option (ex 0, 1, or 2): "))

    option = options[option_index]

    return option
