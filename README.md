
```
{
  "firstName": "John",
  "lastName": "Smith",
  "age": 25
}
```

# 3D-truck-loading
Operations research is a scientific discipline that uses several techniques from fields such as mathematical modeling, simulation, optimization theory, computer science and other fields to solve complex decision making problems concerning processes and operations. In that telling, operations research is linked to decision theory and industrial engineering. Operations research is also known as management science.

The ROADEF/Euro challenge 2022 is a supply chain decision problem. Supply chain decision problems are one of the most famous problems in operations research. They require analytical methods and techniques from a plethora of fields (Queuing theory, Graph theory, Combinatorial Optimization, Discrete modeling...etc).
In this project, few algorithms dedicated to solve the very simple iterations of 3D truck loading problem are presented (Genetic algorithms, Greedy algorithm, Simulated Annealing, Independent Set search, Tabu search...etc), correct the model every step and resolve more complicated versions until the whole problem is covered. Bin packing problems are solved iteratively using different methods, then VM packing problems are considered solved using the same methods but scaled to VM structures. At last, the temporal dimension is added via efficient surjective matching.

# Problem taxonomy
The 3d truck loading problem is an example of packing problem. Packing problems are concerned with packing objects as densely as possible within a single container or packing maximum number of objects in a container as to minimize the number of containers used. There are a lot of variants of packing problems depending on the topology of the objects and the containers.
Packing problems are either geometry-specific or otherwise. The latter case is usually dealt with platonically via abstract data structures. In the case of geometry taken into account, there are two parameters that distinguish between different problems (other than topology or supply collection); boundedness (compact space, Euclidean space..etc) and dimension of the space (3D, 2D..etc).

Complexity-wise, packing problems are NP-hard; meaning that there exists a polynomial-time reduction NP problems of the same nature to packing problems ie that solutions to packing problems can be used to solve NP problems (of the same nature) in polynomial time. However, the fly in the ointment is that polynomial-time algorithms for NP hard problems such as packing problems are not yet proved to exist. In fact, this is equivalent to stating the famous conjecture P = NP.

Packing problems have recovering problems as their dual optimization problems. This will prove useful later in the problem-solving phase. 

# Packing Density: 


Packing density is defined as the fraction of the amount of space filled by objects to the space of the container. Packing density is an important notion in crystallography (known as compacity or atomic packing factor) that we usually choose to maximize (Kepler conjecture, Ulam packing conjecture..etc).\\
The formulation of the packing density is measure-specific (continuous,discrete), space-specific (compact, Euclidean space..etc). For our convenience, trucks are compact (bounded and closed) finite-dimensional (3D) measurable spaces X. Let K1,K2,...,Kn be non-overlapping measurable subsets (stacks in our case).
The packing density of the stacks packing collection is defined as:

η = (∑ᵢ₌₁ⁿ μ(Kᵢ)) / μ(X)

where $\mu$ is the measure defined on our compact space (here taken to be the volume).

Dual to the notion of packing density is the notion of redundant void (redundant because it's not weight-constrained and thus unnecessary void between stacks that need to be exploited by adding more potential stacks in the truck):


RV = 1 - η = (μ(X) - ∑ᵢ₌₁ⁿ μ(Kᵢ)) / μ(X)


Maximizing the packing density (effect) is the inverse problem of the Gibbs entropy minimization (cause). This approach is recommended because there is no explicit geometry-specific formulation of Gibbs entropy. Inverse problems are usually ill-posed (often violating Hadamard's third criterion for well-posed problems; stability/continuity of the solutions with respect to given parameters) which makes it impossible to infer the causes sometimes. However, Gibbs entropy (cause) is just a representative model that fits our goal (a sort of abstract cause, not a physical cause). Moreover, we are not interested in calculating this Gibbs entropy perse from packing density inverse problem. In other words, this is not an either/or situation between forward and inverse problems as both notions are, in our case, completely replaceable. 

# Gibbs entropy vs Conformational entropy

The Gibbs entropy:
refers to the degree of disorder present in a system, which is naturally high in the case of random sampling. This entropy is order-specific and unequally distributed, meaning that some states are more favored than others, as dictated by the second law of thermodynamics. In this context, Gibbs entropy is the measure of weight-unconstrained void that exists between stacks. This void is unnecessary and actually contributes to the high entropy of the system. The analysis assumes that loading constraints have been met, but aims to maximize loading possibilities, or the number of potential future items/stacks to be loaded. This is known as the primal problem of entropy's dual problem. Specifically, the potential loading problem is denoted by K=3, with pre-defined loading problems represented by K=1,2 and the post-loading problem by K=3. This problem can be expressed as the maximization of abs(card(pre-SIs)-card(post-SIs)) using a truck-based approach. The Gibbs entropy is considered the dual of the geometry/packing density/compacity, which is the primal problem.
Combinatorial/Conformational entropy:
refers to the number of ways in which a system, such as a molecule or any thermodynamic system, can be rearranged. Combinatorial entropy is used to enumerate all exhaustive possibilities of 5 degrees of freedom: translation sx,sy,sz (continuously infinite, discretely finite), horizontal rotation, and forced orientation SOs (which reduces from 4 to 2 due to structural symmetry). This includes lengthwise and widthwise rearrangements, which leads to individual void and, eventually, collective void through recursion.
