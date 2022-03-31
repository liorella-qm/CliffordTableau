This is a simple addition to CliffordTableaus which allows us to calculate the 
composition of tableaus and their inverses using the table data directly, and only
elementary arithmetic and logic operations. Thus, cirq is not needed. Dependence 
on numpy was maintained for reasons of convenience. 

The API is WIP and I want to converge to the same API as the original Clifford Tableaus.
The repr is more in line with the mathematical way of working on it using symplectic matrices.

