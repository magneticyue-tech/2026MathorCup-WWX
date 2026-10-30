REPORT_ROWS: tuple[int, ...] = (21, 37, 42, 49, 91, 104, 112, 130, 166, 190, 201, 206)


def multi_date_rows(ratings) -> list[int]:
    n_dates = ratings.groupby("row")["date"].nunique()
    return sorted(int(r) for r in n_dates[n_dates > 1].index)
