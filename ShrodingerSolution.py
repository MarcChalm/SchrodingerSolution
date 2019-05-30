import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



def debugger(list_of_variables, list_of_variable_names):

    for i in range(len(list_of_variables)):

        variable_value = list_of_variables[i]
        variable_name = list_of_variable_names[i]

        print(f'Variable name:{variable_name}, Value:{variable_value}')

# Defining Malfliet-Tjon potential

def potential_V(r):
    # Realistik nukleon-nukleon potential of Malfliet-Tjon type.
    # Ref: R. A. Malfliet and J.A. Tjon
    # Nuclear Physics A127 (1969) 161-168

    # input: Radius r in 10^-15 m or 1 fm
    # Output: Potential V in MeV

    l_1 = -586.04
    l_2 = 1458.19
    l_3 = -872.15

    mu_1 = 1.55
    mu_2 = 3.11
    mu_3 = 6.00

    V_1 = l_1 * np.exp(-mu_1 * r)
    V_2 = l_2 * np.exp(-mu_2 * r)
    V_3 = l_3 * np.exp(-mu_3 * r)

    V = (V_1 + V_2 + V_3) / r

    return V


# Function for populating Vr

def populate_Vr(Vr, r):

    for i in range(Vr.size):

        Vr[i] = potential_V(r[i])

    return Vr

# Function for populating Fvec

def populate_Fvec(Fvec, Vr, E):

    # Mass of protron in MeV/c^2
    mass_protron = 938.272
    # Mass of neutron in MeV/c^2
    mass_neutron = 939.565
    # Reduced mass of the deutron
    my = (mass_protron * mass_neutron) / (mass_protron + mass_neutron)
    # hbar*c = 197.327 MeV*fm
    hbar = 197.327
    # The c cancels out from the MeV/c^2 unit of my
    K = 2 * my / hbar ** 2

    for i in range(Fvec.size):

        Fvec[i] = K * (Vr[i] - E)

    return Fvec


def numerov(u, Fvec, u_0, u_1, index, steplength, revese=False):

    # For when calculating inner integral
    if revese:
        u = u[::-1]

    # Init outward integrated wave function
    u[0] = u_0
    u[1] = u_1
    h = steplength
    # Numerov outward
    for i in range(1, index):
        u_0 = u[i]
        u_neg_1 = u[i - 1]

        F_1 = Fvec[i + 1]
        F_0 = Fvec[i]
        F_neg_1 = Fvec[i - 1]

        numerov_numerator = u_0 * (2 + (5 / 6) * (h ** 2) * F_0) - u_neg_1 * (1 - (1 / 12) * (h ** 2) * F_neg_1)
        numerov_denominator = (1 - (1 / 12) * (h ** 2) * F_1)
        next_step = numerov_numerator / numerov_denominator
        u[i + 1] = next_step

    # For when calculating inner integral
    if revese:
        return u[::-1]

    else:
        return u


def numerov_inner(u, Fvec, u_0, u_1, index, steplength):

    last_index = u.size - 1
    u[last_index] = u_0
    u[last_index - 1] = u_1
    h = steplength

    for i in range(last_index-1, index, -1):

        u_0 = u[i]
        u_neg_1 = u[i + 1]

        F_1 = Fvec[i + 1]
        F_0 = Fvec[i]
        F_neg_1 = Fvec[i - 1]

        # TODO: Flipp all signs? did  it above instead

        numerov_numerator = u_0 * (2 + (5 / 6) * (h ** 2) * F_0) - u_neg_1 * (1 - (1 / 12) * (h ** 2) * F_neg_1)
        numerov_denominator = (1 - (1 / 12) * (h ** 2) * F_1)
        next_step = numerov_numerator / numerov_denominator
        u[i + 1] = next_step
        print(next_step)

    return u


# Declaring constants

# Grid in fm
r_max = 35.0
# Number of steps
N = 35000
# Step lengt
h = r_max / N

print(f'Steplengt: {h}')

# Init grid and potential V(r) for every r
# OBS: Never use 0
r = np.linspace(10.**-16, r_max, num=N)
Vr = populate_Vr(np.zeros(N), r)
u = np.zeros(N)
df = pd.DataFrame()

# Set important parameters
# Guessing whatever was meintioned in the exercise
Emin = min(Vr)
Emax = 0.0
E = 0.5 * (Emin+Emax)
max_iter = 100
continuity_tolerance = 0.00000001
rmp_index = 2500

# Itterate over the energi E

for iter in range(max_iter):


    # Init Fvec(r)
    # This vector is dependent on E
    Fvec = populate_Fvec(np.zeros(N), Vr, E)

    # Choose matching point (equvivalent grid index)
    # In exercise suggested to start at 1fm
    rmp_index = rmp_index

    print(f'Match r={r[rmp_index]}')

    # Init outward integrated wave function
    u_outer = numerov(np.zeros(N), Fvec, 0, h ** 1, rmp_index, h)
    u_out_mp = u_outer[rmp_index]
    df['u_outer'] = u_outer

    # Init inward integrated wave function
    u_inner = numerov(np.zeros(N), Fvec, 0, h ** 1, (N - rmp_index - 2), h, revese=True)
    #u_inner = numerov_inner(np.zeros(N), Fvec, 0, h ** 1, rmp_index, h)
    u_in_mp = u_inner[rmp_index + 1]
    df['u_inner'] = u_inner

    # Correcting u[rmp_index]
    #u[rmp_index] = u[rmp_index] / 2

    # Scaling factor between ingoing and outgoing wave function
    scale_factor = u_out_mp / u_in_mp

    # Match the height and create the full vector u
    u = u_outer + scale_factor * u_inner

    df['u'] = u

    # Calculate the discontinuity of the derivitiv of mp

    # TODO: Is this correct?
    #matching_numerator = (u[rmp_index - 1]) + (u[rmp_index + 1]) - u[rmp_index] * (2 + (h ** 2) * Fvec[rmp_index])
    #matching_denominator = h
    #matching = matching_numerator / matching_denominator

    dx = np.gradient(u, h)
    df['dx'] = dx
    u_outer_dx = dx[rmp_index]
    u_inner_dx = dx[rmp_index + 1]
    matching = (u_inner_dx - u_outer_dx)


    if abs(matching) < continuity_tolerance:
        # Break the loop
        print('Within tolerance!')
        break

    if u[rmp_index] * matching > 0:
        Emax = E

    if u[rmp_index] * matching < 0:
        Emin = E

    # Calculating E for the next iteration.
    E = 0.5 * (Emax + Emin)

# Debugging
print('Testing')
debugger([E, continuity_tolerance, matching], ['E', 'continuity_tolerance', 'matching'])

plt.plot(r[0:20000], u[0:20000])
plt.show()

# After the code works:
# 1. Normalize the function so that the integral = 1
# 2. Calculate the observed radius in fm
# 3. Plot the wave function u(r)
# 4. Analyse the results