
# Clifford algebra with elementary algebra
The goal of this work is to perform Clifford group operations (composition and inversion) using elementary arithmetic and logic operations only. In what follows we develop the formalism and the implementation.

## Pauli algebra using bitvectors
The first thing we'll need is to map Paulis to bitvectors, and to map multiplications between Paulis to operations on these bitvectors with the addition of an overall phase. This can be done by making the one to one mapping $W$ from $\mathbb{F}_2^2$ (binary vectors of length 2, or more formally the 2n-dimensional vector space over the field $\mathbb{F}_2$) to the group of Pauli operators on a single qubit *mod phase*, $\mathcal{P}_1/\mathbb{C}$:
$$
W(0,0)= I \\
W(1,0) = X\\
W(0,1) = Z \\
W(1,1) = Y,
$$
and we map multiplication to addition mod 2. For example, $XY\mapsto(1,0)+(1,1)\mod2=(0,1)\mapsto Z$. 

This can be straighforwadly extended to n qubits by mapping each binary vector of length 2n to an n-qubit Pauli via:
$$
W:\mathbb{F}_2^{2n}\to\mathcal{P}_n/\mathbb{C} \\
$$ 
such that
$$
W(v)=X_0 ^{v_0}Z_0^{v_1}\cdots X_{n-1}^{v_{2n-2}}Z_{n-1}^{v_{2n-1}}.
$$ 
Now if we want to add the phase, we need to keep track of it by working out what the phase is when we multiply two Paulis. This can be easily seen to be given by:
$$
W(v)W(u)=i^{\beta(v,u)}W(v+u)
$$
where
$$
\beta(u,v)=\sum_{k=0}^{n-1}g((v_{2k},v_{2k+1}),(u_{2k},u_{2k+1}))
$$
and $g$ is just the table of the phase we get when we multiply two Paulis together:
| |I|X|Z|Y|
|-|-|-|-|-|
|I|0|0|0|0|
|X|0|0|3|1|
|Z|0|1|0|3
|Y|0|3|1|0|
For example, $(X_0Y_1)(Z_0 X_1)=(X_0Z_0)(Y_1X_1)=(-iY_0)(-iZ_1)=-Y_0Z_1.$

Another thing we'll need is the commutation relation. Two Paulis either commute or anti-commute. This means that:
$$
W(v)W(u) = (-1)^xW(u)W(v),\,x=0,1.
$$
It's easy to see that
$$
x=v^T\Lambda u
$$
where

$\Lambda$  has
$$X=\begin{pmatrix}0 & 1 \\ 1 & 0\end{pmatrix}$$
on the diagonal, namely
$$
\Lambda=\text{diag}(\underbrace{X,..,X}_{\text{n times}}).
$$
This commutation relation is going to be very important in the next section.

## Clifford Algebra using bit-matrices (Tableaus)


The action of a Clifford $C\in \mathcal{C}_n$ (where $\mathcal{C}_n$ is the n qubit Clifford group) on a Pauli $P$ is completely defined by how it maps the "basis" Paulis $X_0,Z_0,\dots X_n,Z_n$. In general, it maps each such basis Pauli to a product of mod-phase Paulis with an overall sign (see [this](http://home.lu.lv/~sd20008/papers/essays/Clifford%20group%20%5Bpaper%5D.pdf)). We denote the corresponding map in the binary vector space as $g$. Thus:
$$
CW(v)C^\dagger=(-1)^{\alpha(v)} W(g(v)).
$$

We can write the mapping explicitly by representing it as a (2n+1)x2n matrix, of the form:
$$
\begin{pmatrix}
g\\\alpha(v_0)\cdots\alpha(v_{2n-1})
\end{pmatrix}.
$$
For example, the identity clifford on 2 qubits looks like this:
```
  |x0 z0 x1 z1
--+------------
x0|1  0  0  0
z0|0  1  0  0
x1|0  0  1  0
z1|0  0  0  1
s |0  0  0  0
```

and the CNOT (0->1) looks like this:

```
   |x0 z0 x1 z1
--+------------
x0|1  0  0  0
z0|0  1  0  1
x1|1  0  1  0
z1|0  0  0  1
s |0  0  0  0
```


### Action of Clifford on a product of Paulis
Let's see what is the action of $C\in\mathcal{C}_n$ on a product of Paulis in terms of the binary space and the phase:
$$
CW(v+u)C^\dagger=\\
i^{2\alpha(v+u)}W(g(v+u))=\\
i^{-\beta(v,u)}CW(v)C^\dagger CW(u)C^\dagger=\\
i^{-\beta(v,u)+2\alpha(v)+2\alpha(u)}W(gv)W(gu)=\\
i^{\beta(gv,gu)-\beta(v,u)+2\alpha(v)+2\alpha(u)}W(gv+gu).
$$
from this we learn two things:
1. $g$ is a linear map, and so is given by a 2nx2n binary matrix over $\mathbb{F}_2$.
2. we have the following addition rule on the signs: $2\alpha(v+u)=\beta(gv,gu)-\beta(v,u)+2\alpha(v)+2\alpha(u)$.

This is all we need to calculate the inverse and the composition of two Cliffords.

### Properties of the linear map $g$

The linear map $g$ is very important. We might naively think that there are $2^{4n^2}$ such matrices, but this is not the case, because the Clifford *must preserve commutation relations between the Paulis*. It's not hard to see that the condition for preserving commutation relations can be expressed as the following condition on $g$:
$$
g^T\Lambda g=\Lambda
$$
where $\Lambda$ was defined above. In the ref above, the number of such $g$'s is explicitly calculated. For 1 qubit there are 6 such matrices, and for 2 qubits there are 720. Since this proporety is closed for multiplication, it means that $g$ forms a group, which is known as the *symplectic group* of 2nx2n matrices over the binary field, $\mathrm{Sp}(2n,\mathbb{F}_2)$. Furthermore, we have:
$$
\mathcal{C}_n/\mathcal{P}_n \simeq \mathrm{Sp}(2n,\mathbb{F}_2).
$$
In the tableau notation, this has a very simple interpretation: $\mathrm{Sp}(2n,\mathbb{F}_2)$ is all the tableau with a zero sign row, and the Paulis are all the tableaus with an identity $g$ and every possible sign row (there are $2^{2n}$ such rows).

### The equation for composing Tableaus

The Clifford Tableau tells us the image of $g_i$, $\alpha_i$ on each one of the basis vectors $e_i$, where $e_0 = x_0$, $e_1 = z_0$, $e_2 = x_1$ and so on. Our goal is to know the image of the composition $C_{21} = C_2C_1$ on each one of the basis vectors.


A composition of two Cliffords is:

$$
C_2^\dagger C_1^\dagger W(v) C_1C_2 = (-1)^{\alpha_1(v)}C_2^\dagger W(g_1v)C_2
= (-1)^{\alpha_1(v) + \alpha_2(g_1v)}W(g_2g_1v).
$$

so:

$$
g_{21} = g_2g_1, \\
\alpha_{21}(e_i) = \alpha_1(e_i) + \alpha_2(g_1e_i).
$$

Since we know how $\alpha_i$ acts on basis vectors, our goal is to express $\alpha_2(g_1e_i)$ as a sum over basis vectors. To do this, we note that $g_1e_i$ is just the i'th column of $g_1$, which can be expressed as a sum of basis vectors. To simplify the notation, we denote:
$$
g_1\equiv g, \\
g_2\equiv h.
$$
Then:
$$
g_1e_i = \sum_{k=0}^{2n-1} g_{ki}e_k \equiv \sum_k g_{ki}e_k.
$$

so we need to calculate

$$
\alpha_2\left(\sum_{k=0}^{2n-1} g_{ki}e_k\right).
$$
TODO: write the calculation. (Uses the equation for sum of $\alpha$).

We end up with:
$$
2\alpha_{21}(e_i) = 2\alpha_1(e_i) + 2\sum_{k=0}^{2n-1}g_{ki}\alpha_2(e_k) + b_i, \mod 4
$$
where $g_{ki}$ is the k'th entry of the i'th column of $g_1$. 

### Inversion of a Clifford operator

we can get the inverse from the equation for the composition above:

$$
2\alpha_{21}(e_i) = 2\alpha_1(e_i) + 2\sum_{k=0}^{2n-1}g_{ki}\alpha_2(e_k) + \\
\sum_{k=0,2,...}^{2n-2}
g_{ki} g_{k+1,i}(1 +\beta(he_k, he_{k+1})) = 0 \mod 4.
$$

In this case, $h=g^{-1}$. 
where $g_2 =g_1^{-1}$. This is easy to calculate because it's a symplectic matrix, i.e. $g\Lambda g^T = \Lambda$, so 
$$
g^{-1} = \Lambda g^T \Lambda
$$

where $\Lambda$ is the symplectic matrix.

Now, if we denote:
$$
a_i = \alpha_1(e_i), \\
b_i = \sum_{k=0,2,...}^{2n-2}
g_{ki} g_{k+1,i}(1 +\beta(\Lambda g^T \Lambda e_k, \Lambda g^T \Lambda e_{k+1})), \\
x_k = \alpha_2(e_k),
$$

we get the equation:
$$
2g^T x = -2a - b \mod 4
$$
so
$$
2x = -\Lambda g \Lambda (2a+b) \mod 4.
$$

