import sys

from crossword import *
from collections import deque


class CrosswordCreator:
    def __init__(self, crossword):
        """
        Create new CSP crossword generate.
        """
        self.crossword = crossword
        self.domains = {
            var: self.crossword.words.copy() for var in self.crossword.variables
        }

    def letter_grid(self, assignment):
        """
        Return 2D array representing a given assignment.
        """
        letters = [
            [None for _ in range(self.crossword.width)]
            for _ in range(self.crossword.height)
        ]
        for variable, word in assignment.items():
            direction = variable.direction
            for k in range(len(word)):
                i = variable.i + (k if direction == Variable.DOWN else 0)
                j = variable.j + (k if direction == Variable.ACROSS else 0)
                letters[i][j] = word[k]
        return letters

    def print(self, assignment):
        """
        Print crossword assignment to the terminal.
        """
        letters = self.letter_grid(assignment)
        for i in range(self.crossword.height):
            for j in range(self.crossword.width):
                if self.crossword.structure[i][j]:
                    print(letters[i][j] or " ", end="")
                else:
                    print("█", end="")
            print()

    def save(self, assignment, filename):
        """
        Save crossword assignment to an image file.
        """
        from PIL import Image, ImageDraw, ImageFont

        cell_size = 100
        cell_border = 2
        interior_size = cell_size - 2 * cell_border
        letters = self.letter_grid(assignment)

        # Create a blank canvas
        img = Image.new(
            "RGBA",
            (self.crossword.width * cell_size, self.crossword.height * cell_size),
            "black",
        )
        font = ImageFont.truetype("assets/fonts/OpenSans-Regular.ttf", 80)
        draw = ImageDraw.Draw(img)

        for i in range(self.crossword.height):
            for j in range(self.crossword.width):

                rect = [
                    (j * cell_size + cell_border, i * cell_size + cell_border),
                    (
                        (j + 1) * cell_size - cell_border,
                        (i + 1) * cell_size - cell_border,
                    ),
                ]
                if self.crossword.structure[i][j]:
                    draw.rectangle(rect, fill="white")
                    if letters[i][j]:
                        _, _, w, h = draw.textbbox((0, 0), letters[i][j], font=font)
                        draw.text(
                            (
                                rect[0][0] + ((interior_size - w) / 2),
                                rect[0][1] + ((interior_size - h) / 2) - 10,
                            ),
                            letters[i][j],
                            fill="black",
                            font=font,
                        )

        img.save(filename)

    def solve(self):
        """
        Enforce node and arc consistency, and then solve the CSP.
        """
        self.enforce_node_consistency()
        self.ac3()
        return self.backtrack(dict())

    def enforce_node_consistency(self):
        """
        Update `self.domains` such that each variable is node-consistent.
        (Remove any values that are inconsistent with a variable's unary
         constraints; in this case, the length of the word.)
        """
        for variable in self.crossword.variables:
            for word in self.domains[variable].copy():  # self.crossword.words:
                if len(word) != variable.length:
                    self.domains[variable].remove(word)

    def revise(self, x, y):
        """
        Make variable `x` arc consistent with variable `y`.
        To do so, remove values from `self.domains[x]` for which there is no
        possible corresponding value for `y` in `self.domains[y]`.

        Return True if a revision was made to the domain of `x`; return
        False if no revision was made.
        for word_x in self.domains[x]:#.copy():
            if all(not self.crossword.consistent(word_x, word_y) for word_y in self.domains[y]):
                self.domains[x].remove(word_x)
                revised = True
        return revised
        """
        revised = False
        index = self.crossword.overlaps.get((x, y))
        if index:
            for word_x in self.domains[x].copy():
                if all(
                    (word_x[index[0]] != word_y[index[1]]) for word_y in self.domains[y]
                ):
                    self.domains[x].remove(word_x)
                    revised = True
        return revised

    def ac3(self, arcs=None):
        """
        Update `self.domains` such that each variable is arc consistent.
        If `arcs` is None, begin with initial list of all arcs in the problem.
        Otherwise, use `arcs` as the initial list of arcs to make consistent.

        Return True if arc consistency is enforced and no domains are empty;
        return False if one or more domains end up empty.
        """
        if arcs is None:
            arcs = deque(
                (x, y)
                for x in self.crossword.variables
                for y in self.crossword.neighbors(x)
            )
        else:
            arcs = deque(arcs)

        while arcs:
            x, y = arcs.popleft()
            if self.revise(x, y):
                if not self.domains[x]:
                    return False
                for z in self.crossword.neighbors(x) - {y}:
                    arcs.appendleft((z, x))
        return True

    def assignment_complete(self, assignment):
        """
        Return True if `assignment` is complete (i.e., assigns a value to each
        crossword variable); return False otherwise.
        """
        for variable in self.crossword.variables:
            if variable not in assignment:
                return False
        return True

    def consistent(self, assignment):
        """
        Return True if `assignment` is consistent (i.e., words fit in crossword
        puzzle without conflicting characters); return False otherwise.
        """
        for variable_x, word_x in assignment.items():
            if variable_x.length != len(word_x):
                return False
            for variable_y, word_y in assignment.items():
                if variable_y != variable_x:
                    if word_y == word_x:
                        return False
                    overlap = self.crossword.overlaps[variable_x, variable_y]
                    if overlap:
                        x = overlap[0]
                        y = overlap[1]
                        if word_x[x] != word_y[y]:
                            return False
        return True

    def order_domain_values(self, var, assignment):
        """
        Return a list of values in the domain of `var`, in order by
        the number of values they rule out for neighboring variables.
        The first value in the list, for example, should be the one
        that rules out the fewest values among the neighbors of `var`.
        """
        neighbors = self.crossword.neighbors(var)  # i first find all a vars neighbors
        var_domain = self.domains[var]  # i want a list of all variables in domain
        order = {word: 0 for word in var_domain}
        # this should hold the key as a word and the value as the eliminations
        for neighbor in neighbors:  # for each of its neighbors
            if neighbor not in assignment:  # if the neighbor hasnt been assigned
                # neighbor_domain = self.domains[neighbor]
                # original_length = len(neighbor_domain)
                overlap = self.crossword.overlaps.get((var, neighbor))
                if overlap:
                    for word in var_domain:
                        for compar in self.domains[neighbor]:  # neighbor_domain.copy():
                            # check if word rules out compar based on overlap
                            # if word == compar:
                            # neighbor_domain.remove(compar)
                            # order[word] += 1
                            # elif overlap:
                            if word[overlap[0]] != compar[overlap[1]]:
                                # neighbor_domain.remove(compar)
                                order[word] += 1
        sorted_order = sorted(order, key=lambda word: order[word])
        return sorted_order

    def select_unassigned_variable(self, assignment):
        """
        Return an unassigned variable not already part of `assignment`.
        Choose the variable with the minimum number of remaining values
        in its domain. If there is a tie, choose the variable with the highest
        degree. If there is a tie, any of the tied variables are acceptable
        return values.
        """
        
        variables = self.crossword.variables
        unassigned = [var for var in variables if var not in assignment]

        def mrv_and_degree(var):
            remaining_values = len(self.domains[var])
            degree = len(self.crossword.neighbors(var))
            return (remaining_values, degree)

        selected = min(
            unassigned,
            key=lambda var: (mrv_and_degree(var)[0], -mrv_and_degree(var)[1]),
        )
        return selected


    def infer(self, assignment):
        # after making a new assigmnwnt x call the ac3 algo and start it
        # with a queue o all arcs (y,x) where y is a neighbor of x.
        inferences = {}
        for x in assignment:
            arcs = [(x, y) for y in self.crossword.neighbors(x)]
            if not self.ac3(arcs):
                return None  # return early if arc consistency fails
            # add inferences made
            for variable in self.domains:
                if len(self.domains[variable]) == 1 and variable not in assignment:
                    inferences[variable] = next(iter(self.domains[variable]))
        return inferences

    def backtrack(self, assignment):
        """
        Using Backtracking Search, take as input a partial assignment for the
        crossword and return a complete assignment if possible to do so.

        `assignment` is a mapping from variables (keys) to words (values).

        If no assignment is possible, return None.
        """

        if self.assignment_complete(assignment):
            return assignment
        variable = self.select_unassigned_variable(
            assignment
        )  # we select an unassigned var
        for value in self.order_domain_values(variable, assignment):
            mock_assignment = assignment.copy()
            mock_assignment[variable] = value
            if self.consistent(
                mock_assignment
            ):  # if variable doesnt violate any constraints
                assignment[variable] = value  # we add the assignment

                inferences = self.infer(assignment)

                if inferences != None:
                    # add inferences to assignment
                    for var, value in inferences.items():
                        assignment[var] = value

                result = self.backtrack(assignment)
                if result != None:
                    return result
                assignment.pop(variable)

                # remove inferences from assignment
                for var in inferences:
                    del assignment[var]

        return None


def main():

    # Check usage
    if len(sys.argv) not in [3, 4]:
        sys.exit("Usage: python generate.py structure words [output]")

    # Parse command-line arguments
    structure = sys.argv[1]
    words = sys.argv[2]
    output = sys.argv[3] if len(sys.argv) == 4 else None

    # Generate crossword
    crossword = Crossword(structure, words)
    creator = CrosswordCreator(crossword)
    assignment = creator.solve()

    # Print result
    if assignment is None:
        print("No solution.")
    else:
        creator.print(assignment)
        if output:
            creator.save(assignment, output)


if __name__ == "__main__":
    main()
