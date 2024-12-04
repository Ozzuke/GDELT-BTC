def calc_event_magnitude(row):
    return round(
        (
            row['NumArticles'] +
            10 * row['NumSources'] +
            20 * abs(row['AvgTone'])
        ) / 20, 2)

def calc_event_impact(row):
    return round(
        (
            row['Magnitude'] *
            row['GoldsteinScale']
         ) / 10, 2)

def get_impact_bin(impact):
    if impact < -5:
        return 'Very Negative'
    elif impact < -2:
        return 'Negative'
    elif impact < -0.5:
        return 'Slightly Negative'
    elif impact < 0.5:
        return 'Neutral'
    elif impact < 2:
        return 'Slightly Positive'
    elif impact < 5:
        return 'Positive'
    else:
        return 'Very Positive'
