from shopping import load_data

csv_file = "testcsv.csv"
test_evidence = [0, 0.0, 0, 0.0, 1, 0.0, 0.2, 0.2, 0.0, 0.0, 1, 1, 1, 1, 1, 1, 0]
def test_loader():
    evidence, labels = load_data(csv_file)

    assert evidence[0] == test_evidence
    assert labels == [0]
    assert len(evidence[0]) == len(test_evidence)