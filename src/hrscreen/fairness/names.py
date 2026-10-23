"""First names used for the counterfactual (name-swap) audit.

These are gender-CODED names, i.e. names that most people would read as female or male. They are
a stimulus for testing whether a ranker reacts to a name; they are not a way to determine any
real person's gender. Surnames are shared between the two variants of a resume, so only the
first name changes.
"""

from __future__ import annotations

import numpy as np

FEMALE_CODED = [
    "Priya", "Ananya", "Sneha", "Pooja", "Kavita", "Neha", "Meera", "Divya", "Sunita", "Lakshmi",
    "Fatima", "Aisha", "Emily", "Sarah", "Jessica", "Emma", "Olivia", "Hannah",
]
MALE_CODED = [
    "Rahul", "Amit", "Vikram", "Rohan", "Suresh", "Arjun", "Karthik", "Imran", "Rajesh", "Manoj",
    "Sanjay", "Mohammed", "James", "John", "Michael", "David", "Daniel", "Thomas",
]
SURNAMES = [
    "Sharma", "Patel", "Singh", "Kumar", "Mehta", "Iyer", "Khan", "Reddy", "Verma", "Nair",
    "Smith", "Johnson", "Brown", "Williams", "Taylor", "Davies",
]


def sample_name_pairs(n: int, rng: np.random.Generator) -> list[tuple[str, str]]:
    """``n`` (female-coded full name, male-coded full name) pairs that share a surname."""
    pairs = []
    for _ in range(n):
        surname = str(rng.choice(SURNAMES))
        pairs.append((f"{rng.choice(FEMALE_CODED)} {surname}", f"{rng.choice(MALE_CODED)} {surname}"))
    return pairs
