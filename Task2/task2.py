# Defining a Node class with elements as the value it contains, the nodes from which it got is value (childrenNodes, nodeA, nodeB)
# The operation which operates on children nodes, that is value = nodeA op nodeB and some additional operations (for now none) 
class Node:
    # Defining the constructor
    def __init__(self, value=None, children=None, op=None, op_args=None):
        self.value = value
        # If there is no value associated, it means value will come from previous nodes
        # If there is already a value, there is no meaning of having children Nodes
        if value == None:
            # If value is None, then children nodes are inputs, if input itself is empty then list if empty
            # (Technically a useless node but whatever)
            self.children = children if children != None else []
        else:
            # If value is something, then children nodes are not there, hence the list is empty
            self.children = []
        self.op = op
        self.op_args = op_args

    # Function to evaluate what the value of the Node should be
    def evalVal(self):
        # If children nodes are not there then, the value taken as input is the value it self
        # If the operation is not +, -, *, /, ** then value is none
        if self.children == [] or self.op not in ["add", "sub", "mul", "div", "exp"]:
            return self.value

        # Defining what to do when the operation is add, sub, mul, div or exp
        if self.op == "add":
            self.value = self.children[0].value + self.children[1].value

        # Note if children node are [A, B] and Node.val = k, then if children node are [B, A] then Node.val = -k
        elif self.op == "sub":
            self.value = self.children[0].value - self.children[1].value

        elif self.op == "mul":
            self.value = self.children[0].value * self.children[1].value

        # Note if children node are [A, B] and Node.val = k, then if children node are [B, A] then Node.val = 1/k
        elif self.op == "div":
            if self.children[1].value != 0:
                self.value = self.children[0].value / self.children[1].value
            else:
                self.value = float('nan')  # Handle division by zero

        # Have to give some if conditions for some cases like (-1) ** (1/2), we will add later 
        elif self.op == "exp":
            self.value = self.children[0].value ** self.children[1].value

        return self.value

    # A function which displays the Node's content in a proper format
    def __repr__(self):
        if self.children:
            return f"Node(value={self.value}, op={self.op}, children=[Node({self.children[0].value}), Node({self.children[1].value})])"
        else:
            return f"Node(value={self.value}, op={self.op}, children=[None])"

# Example usage
Node1 = Node(1)
Node2 = Node(2)
Node4 = Node(5)
Node3 = Node(None, [Node1, Node2], "add", None)
Node5 = Node(None, [Node3, Node4], "mul", None)
# Evaluate the value of Node3, Node5
Node3.evalVal()
Node5.evalVal()
# Print the node to see the result
print(Node3)  # Output: Node(value=15, op=mul, children=[Node(3), Node(5)])

# One of the downsides of this code is that we have calculate the values at all children nodes before we calculate the value
# of the Current node, tbh idk if this is downside, for now I will just leave it as it is and comeback later