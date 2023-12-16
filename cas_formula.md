### Symbols ###
| Symbol           | Description                                    |
|------------------|------------------------------------------------|
| $p$ | static ambient pressure |
| $p_0$ | sea level standard atmosphere pressure |
| $p_t$ | total or stagnation pressure at mouth of pitot tube |
| $p_{t1}$ | for supersonic flow, the total pressure aft of the shock |
| $\Delta p$ or $q_c$ | impact or dynamic pressure |
| $\rho$ | air density |
| $\gamma$ | ratio of specific heats aka adiabatic index; $1.400$ for standard atmosphere dry air |
| $a$ | ambient speed of sound |
| $h$ | geopotential altitude above mean sea level; altitude as measured by altimeter; altimeter calibration is based on standard atmosphere for $p$ |
| $a_0$ | sea level standard atmosphere speed of sound |
| $V_c$ | calibrated air speed |
| $V_t$ | true air speed |

### Formula ###
| Formula                 | Eq. | Description                     |
|-------------------------|--------|---------------------------------|
| $p = \rho R T$ | (1) | Equation of state for atmosphere |
| $\Delta p = p_t - p$ rearranged $p_t = \Delta p + p$ | (2) | pressure difference aka impact pressure which is what is measured by the airspeed indicator; see Eq. (16) for Mach meter equivalent |
| $\frac{\gamma}{\gamma-1}\frac{p}{\rho}+\frac{1}{2}V^2 = constant = \frac{\gamma}{\gamma-1}\frac{p_t}{\rho_t}$ | (3) | from Bernouili's equation for compressible fluid flow; assumes air brought to rest at the stagnation point, where $V=0$ adiabatically; for subsonic speeds, stagation conditions are reached isentropically |
| $\frac{\rho_t}{\rho}=(\frac{p_t}{p})^{1/\gamma}$ | (4) | for subsonic speeds where stagnation is reached isentropically |
| $p_t/p = \Delta p/p+1$ | (5) | note for reference |
| $V^2=\frac{2 \gamma p}{(\gamma-1)\rho}[(\frac{p_t}{p})^{(\gamma-1)/\gamma}-1]$ | (6) | Solving for $V^2$ using Eq. (2) |
| $a^2 = \frac{\gamma p}{\rho}$ | (7) | Can be show speed of sound related to pressure and density |
| $a^2 = \gamma R T$ | (8) | Combining Eq. (1) and (7) |
| $V^2=\frac{2 a^2}{\gamma-1}[(\frac{\Delta p}{p}+1)^{(\gamma-1)/\gamma}-1]$ | (9) | Using Eq. (4) and (6); applies for subsonic speeds |
| $V_c = V$ | (10) | at sea level with standard atmosphere i.e. $a=a_0$ and $p=p_0$ |
| $V_c^2=\frac{2 a_0^2}{\gamma-1}[(\frac{\Delta p}{p_0}+1)^{(\gamma-1)/\gamma}-1]$ | (11) | From Eq. (9) with airspeed instruments calibrated using constant values for $p$ and $a$, namely standard values at sea level |
| $V_c = V_i$ | (12) | taken as assumed for this analysis that indicated airspeed is precisely equal to calibrated airspeed |
| $M=\frac{V}{a}$ | (13) | definition of mach number |
| $M^2=\frac{2}{\gamma-1}[(\frac{\Delta p}{p}+1)^{(\gamma-1)/\gamma}-1]$ | (14) | from Eq. (9) and (13) we have the Saint-Venant formula; the calibration formula for Machmeters at subsonic speeds, with ${\Delta p}/{p_0}$ being measured directly by the Pitot-Machmeter system |
| $\frac{\Delta p}{p}=(1+0.2 M^2)^{3.5} - 1$ | (15) | Eq. (14) rearranged and simplified for $\gamma = 1.400$ |
| $\Delta p = p_{t1} - p$ | (16) | pressure difference measured by the Pitot-Machmeter system |
| $p_{t1} = p(\frac{\gamma + 1}{2})^{(\gamma + 1)/(\gamma - 1)} \bullet (\frac{2}{\gamma - 1})^{1/(\gamma - 1)} \bullet \frac{M^{2 \gamma / (\gamma - 1)}}{(\frac{2 \gamma}{\gamma - 1} \bullet M^2 - 1)^{1/(\gamma - 1)}}$ | (17) | relationship for supersonic speeds |
| $\frac{\Delta p}{p} = \frac{166.92158 M^7}{(7 M^2 - 1)^{2.5}} - 1$ | (18) | Eq. (17) for $\gamma = 1.400$; the Lord Raleigh formula; used when $\Delta p / p \gt 0.893$; use Eq. (15) otherwise |
| $\frac{\Delta p}{p_0} = [1 + 0.2(\frac{V_c}{a_0})^2]^{3.5} - 1$ | (19) | for subsonic speeds and $\gamma = 1.400$; rearrangement of Eq. (11) |
| $\frac{\Delta p}{p_0} = \frac{166.92158 (V_c/a_0)^7}{[7 (V_c/a_0)^2 - 1]^{2.5}} - 1$ | (20) | for supersonic speeds and $\gamma = 1.400$ |

### Calculation of Calibrated Air Speed and Mach number ###
#### Example 1 ####
For $h=36,089 ft$, $p/p_0 = 0.223361$ with $p_0 = 1.01325 \times 10^5 N/m^2$, for a known $Vc$ and $a_0$ $\Delta p$ can be calculated using Eq. (15) for subsonic speeds and (18) for supersonic.

| $V_c (kt)$ | 200 | 400 | 600 | 800 | 1000 |
|------------|-----|-----|-----|-----|------|
| $\Delta p (N/m^2)$ | 6634 | 28395 | 71369 | 145406 | 249057 |

With $\Delta p$ known, the Mach number $M$ can be calculated from Eq. (15) for subsonic speeds and (18) for supersonic speeds with input of the static pressure $p$.

In the standard atmosphere the temperature, speed of sound, pressure and density can all be calculated as follows.

| Parameter | $h \lt 11,000m$ | $11,000m \leq h \lt 20,000m$ |
|-----------|-----------------|-------------------------|
| Temperature | $\frac{T}{T_0} = 1 - \frac{6.5 \times 10^{-3}}{288.150}h$ | $T=T_*=216.650 deg K$ |
| Speed of sound | $\frac{a}{a_0} = (\frac{T}{T_0})^{0.5}$ | $a=a_*=295.070 m/s$ |
| Pressure | $\frac{p}{p_0} = (\frac{T}{T_0})^{5.2558774}$ | $\frac{p}{p_0} = 0.223361 \times exp(-\frac{h-11000}{6341.6184})$ |
| Density | $\frac{\rho}{\rho_0} = (\frac{T}{T_0})^{4.2558774}$ | $\frac{\rho}{\rho_0} = 0.297076 \times exp(-\frac{h-11000}{6341.6184})$ |

#### Example 2 ####
For $h=11,000m$, then $p = 22632 N/m^2$ and $a=295.070 m/s$

| Parameter | $V_c=200 kt$ | $Vc=400 kt$ |
|-----------|--------------|-------------|
| $\Delta p$ | $6634 N/m^2$ | $28395 N/m^2$ |
| $\frac{\Delta p}{p}$ | $\frac{6634}{22632}=0.293125$ | $\frac{28395}{22632}=1.254639$ |
| $M$ | $0.617$ using Eq. (15) | $1.1458$ using Eq. (18) since $\frac{\Delta p}{p} \gt 0.893$ |
| $V$ | $182.058 m/s$ | $338.09 m/s$ |

#### General Algorithim for Calculating Mach number from Calibrated Air Speed ####
1. Get $Vc$, $T$ and $h$ for current conditions
2. Calculate $a$ and $p$ based on above
3. Use Eq. (19) to calculate $\Delta p/p_0$; if result is $\gt 0.893$ then use result from Eq. (20)
4. Multiply by $p_0$ to get $\Delta p$
5. Calculate $p_{ratio} = \Delta p/p$
6. If result from step 5 is $\gt 0.893$ then use Eq. (18) to calculate $M$; else use Eq. (15)
7. To solve step 6 we need to iterate on values of $M$ to minimize $abs(p_{ratio} - [M_{15},M_{18}])$ where $[M_{15},M_{18}]$ are the outputs of Eq. (15) or (18) as appropiate

#### General Algorithim for Calculating Calibrated Air Speed from Mach number from ####
1. Get $M$, $T$ and $h$ for current conditions
2. Calculate $a$ and $p$ based on above
3. Use Eq. (15) to calculate $\Delta p/p$; if result is $\gt 0.893$ then use result from Eq. (18)
4. Multiply by $p$ to get $\Delta p$
5. Calculate $p_{ratio,0} = \Delta p / p_0$
6. If result from step 5 is $\gt 0.893$ then use Eq. (20) to calculate $Vc$; else use Eq. (19)
7. To solve step 6 we need to iterate on values of $V_c$ to minimize $abs(p_{ratio} - [V_{c,19},V_{c,20}])$ where $[V_{c,19},V_{c,20}]$ are the outputs of Eq. (19) or (20) as appropiate