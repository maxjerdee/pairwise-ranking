from ranking import parsing, sampling

match_list = parsing.read_match_list_from_gml("data/gml_files/dogs.gml")

print(sampling.samples(match_list))

print(sampling.samples(match_list))
