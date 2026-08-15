import random

def generate_ticket():
    ticket = [[0 for _ in range(9)] for _ in range(3)]

    columns = [
        list(range(1, 10)),
        list(range(10, 20)),
        list(range(20, 30)),
        list(range(30, 40)),
        list(range(40, 50)),
        list(range(50, 60)),
        list(range(60, 70)),
        list(range(70, 80)),
        list(range(80, 91))
    ]

    positions = []

    for row in range(3):
        positions.append(random.sample(range(9), 5))

    while len(set(positions[0] + positions[1] + positions[2])) < 9:
        positions = [random.sample(range(9), 5) for _ in range(3)]

    for col in range(9):
        rows = [r for r in range(3) if col in positions[r]]
        nums = sorted(random.sample(columns[col], len(rows)))

        for row, num in zip(rows, nums):
            ticket[row][col] = num

    return ticket

def print_ticket(ticket):
    print("+----" * 9 + "+")
    for row in ticket:
        print("|" + "|".join(f"{n:3}" if n else "   " for n in row) + "|")
        print("+----" * 9 + "+")

for i in range(5):
    print(f"\nTicket {i + 1}")
    print_ticket(generate_ticket())