"""
Simple Python script to parse the names files.
"""
import glob
import operator
import cPickle
import numpy as np

def parse_names_files():

    state_to_names_dict = {}

    for fname in glob.glob("*.TXT"):
        state = fname[:2]
        print "Parsing state: ", state
        male_names, female_names = parse_state(fname)

        state_to_names_dict[state] = male_names, female_names

    # Get the sorted states
    states = sorted(state_to_names_dict.keys())

    # Get the top 100 names
    nnames = 100
    top_male_names, top_female_names = get_top_names(state_to_names_dict, N=nnames)

    print "Top male names: "
    print top_male_names
    print "Top female names: "
    print top_female_names

    # Now get a year x state x name array of the top counts
    nyears  = 2013 - 1910 + 1
    years = np.arange(1910, 2014)
    nstates = 51
    male_year_state_names   = np.zeros((nyears, nstates, nnames))
    female_year_state_names = np.zeros((nyears, nstates, nnames))

    for i,state in enumerate(states):
        for j,male_name in enumerate(top_male_names):
            try:
                male_year_state_names[:,i,j] = state_to_names_dict[state][0][male_name]
            except:
                # Name must not have been given in that state
                print "No ", male_name, "\'s found in the state of ", state, " in the years 1910-2013!"

    for i,state in enumerate(states):
        for j,female_name in enumerate(top_female_names):
            try:
                female_year_state_names[:,i,j] = state_to_names_dict[state][1][female_name]
            except:
                # Name must not have been given in that state
                print "No ", female_name, "\'s found in the state of ", state, " in the years 1910-2013!"

    with open('male_data.pkl', 'w') as f:
        cPickle.dump((male_year_state_names, years, states, top_male_names), f, protocol=-1)
    with open('female_data.pkl', 'w') as f:
        cPickle.dump((female_year_state_names, years, states, top_female_names), f, protocol=-1)


def parse_state(fname, end_year=2013):
    """
    Each line of the file is of the form:
        LA,F,1910,Mary,586

    :param fname:
    :return:
    """
    nyears = year_to_index(end_year) + 1
    male_names = {}
    female_names = {}
    lineno = 0
    with open(fname, 'r') as f:
        for line in f:
            lineno += 1

            # if lineno % 1000 == 0:
            #     print "\tLine ", lineno

            # Split the line
            state, gender, year, name, count = line.split(',')
            # Select the dictionary based on gender
            if gender.lower() == 'm':
                D = male_names
            else:
                D = female_names
            name = name.lower()
            index = year_to_index(int(year))
            # Update the count for this particular name
            try:
                D[name][index] = int(count)
            except:
                # Name is not in dict. Add array and increase
                D[name] = np.zeros(nyears)
                D[name][index] = int(count)

    return male_names, female_names


def get_top_names(state_to_names_dicts, N=100):
    """
    Get the top N names from all states for all time
    :param state_to_names_dicts:
    :return:
    """
    male_name_counts = {}
    female_name_counts = {}
    for state,(male_names, female_names) in state_to_names_dicts.items():
        print "Counting names for state: ", state
        for male_name,counts in male_names.items():
            try:
                male_name_counts[male_name] += counts.sum()
            except:
                # Name is not in the database
                male_name_counts[male_name] = counts.sum()

        # Do the same for female names
        for female_name,counts in female_names.items():
            try:
                female_name_counts[female_name] += counts.sum()
            except:
                # Name is not in the database
                female_name_counts[female_name] = counts.sum()


    # Now get the top N names for each
    sorted_male_names   = sorted(male_name_counts.items(), key=operator.itemgetter(1))
    sorted_female_names = sorted(female_name_counts.items(), key=operator.itemgetter(1))

    top_male_names   = [name for (name, count) in sorted_male_names[-N:]][::-1]
    top_female_names = [name for (name, count) in sorted_female_names[-N:]][::-1]

    return top_male_names, top_female_names

def year_to_index(year):
    return year - 1910

parse_names_files()