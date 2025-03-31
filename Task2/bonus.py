import math

class Node:
    def __init__(self, value=None, children=None, op=None, op_args=None):
        self.value = value
        if value == None:
            self.children = children if children != None else []
        else:
            self.children = []
        self.op = op
        self.op_args = op_args

    def evalVal(self):
        if self.children == [] or self.op not in ["add", "sub", "mul", "div", "exp"]:
            return self.value
        if self.op == "add":
            self.value = self.children[0].value + self.children[1].value
        elif self.op == "sub":
            self.value = self.children[0].value - self.children[1].value
        elif self.op == "mul":
            self.value = self.children[0].value * self.children[1].value
        elif self.op == "div":
            if self.children[1].value != 0:
                self.value = self.children[0].value / self.children[1].value
            else:
                self.value = float('nan')
        elif self.op == "exp":
            self.value = self.children[0].value ** self.children[1].value
        return self.value

    def __repr__(self):
        if self.children:
            return f"Node(value={self.value}, op={self.op}, children=[Node({self.children[0].value}), Node({self.children[1].value})])"
        else:
            return f"Node(value={self.value}, op={self.op}, children=[None])"

Node1 = Node(1)
Node2 = Node(2)
Node4 = Node(5)
Node3 = Node(None, [Node1, Node2], "sub", None)
Node5 = Node(None, [Node3, Node4], "exp", None) 

Node3.evalVal()
Node5.evalVal()

"""
Let the final node (Top node, is not a child) represent the output function f,
and base node (Have no children) represent independent variables that is f = f(x,y,z) lets say
Now we have to calculate the gradient of f at some particular Values x0, y0, z0
The gradient at each node represent the Gradient of the output function with respect to the function at that particular node
Case 0 Grad at Top node:
    as df/df = 1, the gradient at top nodes is 1
Case 1 Addition:
    For any node which represent function g (the main function is still f) if g = g1 + g2 (g1 and g2 are functions in Children node)
    df/dg1 = (df/dg) * (dg/dg1) = df/dg similarly, df/dg2 = df/dg
    So if a node perform addition then the grad of the children node is same as the Gradient of parent node
Case 2 Subtraction:
    Now let g = g1 - g2 then df/dg1 = df/dg, df/dg2 = -df/dg
    So if a node perform subtraction the gradient of first child node will be same but grad of second child node is negative of parent
Case 3 Multiplication:
    Let g = g1 * g2 then df/dg1 = g2 * (df/dg), df/dg2 = g1 * (df/dg)
    If a node perform Mult, grad of first child is grad of parent * value of 2nd child and vice versa
Case 4 Division:
    Let g = g1 / g2 then df/dg1 = (1/g2) * (df/dg), df/dg2 = (df/dg) * (-g1/(g2**2))
    Given g2 != 0 (Already took care in class)
Case 5 Exponentiation:
    Let g = g1 ** g2 then df/dg1 = (df/dg) * (g2) * (g1 ** (g2 - 1)), df/dg2 = (df/dg) * (g) * ln(g2) give g2 > 0
"""

def grad(CalcNode):
    """
    Compute gradients for all nodes.
    Returns the gradient for CalcNode with respect to the final output.
    """
    # Our list of all nodes in the computation graph.
    AllNodes = [Node1, Node2, Node4, Node3, Node5]

    # Build a set of all nodes that appear as a child.
    childrenNodes = set()
    for node in AllNodes:
        for child in node.children:
            childrenNodes.add(child)

    # Top nodes are those that are not children of any node.
    TopNodes = [node for node in AllNodes if node not in childrenNodes]

    # We will create new attribute grad
    # Initialize gradient for each node as 0.
    for node in AllNodes:
        node.grad = 0
    # For top (output) nodes, the gradient is 1.
    for node in TopNodes:
        node.grad = 1

    # Process nodes in reverse order (assumed valid order: Base nodes first, then intermediates, then Top nodes).
    # In our example: reverse of [Node1, Node2, Node3, Node4, Node5] is [Node5, Node4, Node3, Node2, Node1]
    for node in reversed(AllNodes):
        # Skip Base nodes and Top nodes
        if node.op is None or not node.children:
            continue

        if node.op == "add":
            # For z = a + b, dz/da = 1 and dz/db = 1.
            node.children[0].grad += node.grad * 1
            node.children[1].grad += node.grad * 1

        elif node.op == "sub":
            # For z = a - b, dz/da = 1 and dz/db = -1.
            node.children[0].grad += node.grad * 1
            node.children[1].grad += node.grad * -1

        elif node.op == "mul":
            # For z = a * b, dz/da = b and dz/db = a.
            node.children[0].grad += node.grad * node.children[1].value
            node.children[1].grad += node.grad * node.children[0].value

        elif node.op == "div":
            # For z = a / b, dz/da = 1/b and dz/db = -a/(b^2)
            if node.children[1].value != 0:
                node.children[0].grad += node.grad * (1 / node.children[1].value)
                node.children[1].grad += node.grad * (-node.children[0].value / (node.children[1].value ** 2))

        elif node.op == "exp":
            # For z = a^b:
            # dz/da = ba^(b-1)  
            # dz/db = (a^b)ln(a) (if a > 0)
            a = node.children[0].value
            b = node.children[1].value
            node.children[0].grad += node.grad * (b * (a ** (b - 1)))
            if a != 0:
                node.children[1].grad += node.grad * (node.value * math.log(math.fabs(a)))
            else:
                None
    
    # Return the computed gradient for the requested node.
    return CalcNode.grad


# --- Example Usage of grad() ---
print("Gradient at Node5 (output):", grad(Node5))  # Expected: 1 (by definition)
print("Gradient at Node3 (intermediate):", grad(Node3))  # Expected: derivative of Node5 w.r.t Node3: Node4.value (which is 5)
print("Gradient at Node1 (base):", grad(Node1))  # Expected: derivative through Node3: 1 * Node4.value = 5
print("Gradient at Node2 (base):", grad(Node2))  # Expected: same as Node1, 5
print("Gradient at Node4 (base):", grad(Node4))  # Expected: derivative of Node5 w.r.t Node4: Node3.value (which is 3)