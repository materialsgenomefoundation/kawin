'''
Databases for unit testing
'''

ALZR_TDB = """
$Al-Zr database
$From T. Wang, Z. Jin, J. Zhao, Journal of Phase Equilibria, 22 (2001) p. 544
$
TEMP_LIM 298.15 6000 !
$Element     Standard state       mass [g/mol]     H_298      S_298
ELEMENT  /-   ELECTRON_GAS        0.0             0.0         0.0 !
ELEMENT  VA   VACUUM              0.0000E+00      0.0000E+00  0.0000E+00 !
ELEMENT  AL   FCC_A1              2.6982E+01      4.5773E+03  2.8322E+01 !
ELEMENT  ZR   HCP_A3              9.1224E+01      5.5663E+03  3.9181E+01 !
$
$
 TYPE_DEFINITION % SEQ *!
$
$PHASE AL3ZR
PHASE AL3ZR % 2 0.75 0.25 !
CONST AL3ZR : AL : ZR : !
$
$PHASE FCC_A1
PHASE FCC_A1 % 2 1 1 !
CONST FCC_A1 : AL%,ZR : VA : !
$
$
$
$
$UNARY DATA
$
$AL (FCC_A1)
FUNCTION GHSERAL   298.15 -7976.15+137.093038*T-24.3671976*T*LOG(T)
             -1.884662E-3*T**2-0.877664E-6*T**3+74092*T**(-1);
         700.00 Y -11276.24+223.048446*T-38.5844296*T*LOG(T)
              +18.531982E-3*T**2-5.764227E-6*T**3+74092*T**(-1);
         933.47 Y -11278.378+188.684153*T-31.748192*T*LOG(T)
                              -1230.524E25*T**(-9);  2900.00 N !
$
$ ZIRCONIUM (GHSERZR FOR HCP_A3)
$
FUNCTION GHSERZR 130.00 -7827.595+125.64905*T-24.1618*T*LOG(T)
                            -4.37791E-3*T**2+34971*T**(-1);
            2128.00 Y -26085.921+262.724183*T-42.144*T*LOG(T)
                            -1342.895E28*T**(-9); 6000.00 N !

$
$                                                                       PHASE FCC_A1
$
PARAMETER G(FCC_A1,AL:VA;0) 298.15 GHSERAL; 6000.00 N !
PARAMETER G(FCC_A1,ZR:VA;0) 298.15 7600.00-0.9*T+GHSERZR; 6000.00 N !
PARAMETER G(FCC_A1,AL,ZR:VA;0)  298.15 -152947+21.3*T;    6000.00  N !

$
$                                                                       PHASE AL3ZR
$
PARAMETER G(AL3ZR,AL:ZR;0)  298.15 -47381 - 24.373*T + 3.894*T*LOG(T)
                              +0.75*GHSERAL+0.25*GHSERZR;  6000.00  N !

$
$                                                                       Diffusion parameters
$
PARAMETER DF(FCC_A1&ZR,*:VA;0) 298.15 -2.56655 * 8.314 * T; 6000 N !
PARAMETER DQ(FCC_A1&ZR,*:VA;0) 298.15 -242000; 6000 N !
"""

NICRAL_TDB = """

$ The parameters of the following database follows the publication
$ N. Dupin, I. Ansara, Thermodynamic Re-Assessment of the Ternary System
$ Al-Cr-Ni, Calphad, 25 (2), 279-298 (2001).
$

$Element     Standard state             mass [g/mol]    H_298       S_298
 ELEMENT /-   ELECTRON_GAS               .0000E+00       .0000E+00   .0000E+00!
 ELEMENT VA   VACUUM                     .0000E+00       .0000E+00   .0000E+00!
 ELEMENT AL   FCC_A1                    2.6982E+01      4.5773E+03  2.8322E+01!
 ELEMENT CR   BCC_A2                    5.1996E+01      4.0500E+03  2.3560E+01!
 ELEMENT NI   FCC_A1                    5.8690E+01      4.7870E+03  2.9796E+01!

 SPECIES AL2                         AL2!
 SPECIES CR2                         CR2!
 SPECIES NI2                         NI2!

 
 FUNCTION UNASS      298.15  0;,,N !


 TYPE_DEFINITION % SEQ *!
 DEFINE_SYSTEM_DEFAULT E 2 !
 DEFAULT_COMMAND DEF_SYS_ELEMENT VA !
 DEFAULT_COMMAND REJECT_PHASE NEWSIGMA !

 DATABASE_INFO '''
 BASE TC-Ni, version 29-09-99'

 '
 ELEMENTS : Al, Cr, Ni'
 '
 ASSESSED SYSTEMS :'
 '
 BINARIES'
        Al-Cr, Al-Ni, Cr-Ni'
 TERNARIES'
        Al-Cr-Ni'
'
 MODELLING ORDER/DISORDER:'
'
 A1 and L12 phases are modelled with a single Gibbs energy curve.'
 They are FCC_L12#1 (A1) based on (Ni) and FCC_L12#2 (L12) based on'
 Ni3Al, differing by their site occupation.'
 The same type of relation exists for the A2 and B2 phases. There are'
 several possible sets for the phase named BCC_B2. They are either'
 disordered (A2) and correspond to the solid solution based on Cr,'
 or ordered based on the B2 compound AlNi.'
 !

ASSESSED_SYSTEMS
 AL-CR(;G5 MAJ:BCC_B2/CR:CR:VA ;P3 STP:.75/1200/1)
 AL-NI(;P3 STP:.75/1200/1)
 CR-NI(;G5 MAJ:BCC_B2/CR:CR:VA C_S:BCC_B2/NI:NI:VA
          ;P3 STP:.5/1200/2)
              !
			  
 TYPE_DEFINITION A GES A_P_D FCC_L12 MAGNETIC  -3.0 .28 !
 TYPE_DEFINITION E GES A_P_D FCC_A1 MAGNETIC  -3.0 .28 !
 TYPE_DEFINITION B GES A_P_D BCC_B2 MAGNETIC  -1.0 .40 !
 TYPE_DEFINITION F GES A_P_D BCC_A2 MAGNETIC  -1.0 .40 !

 TYPE_DEFINITION C GES A_P_D BCC_B2 DIS_PART BCC_A2 !
 TYPE_DEFINITION D GES A_P_D FCC_L12 DIS_PART FCC_A1 !

$ TYPE_DEFINITION G GES A_P_D FCC_L12 C_S 2 NI:AL:VA !
$ TYPE_DEFINITION G GES A_P_D FCC_L12 MAJ 1 NI:NI:VA !
$ TYPE_DEFINITION W GES A_P_D BCC_B2 C_S,, NI:AL:VA !
$ TYPE_DEFINITION W GES A_P_D BCC_B2 MAJ 1 CR:CR:VA !

 PHASE GAS:G %  1  1.0  !
 CONST GAS:G :AL,AL2,CR,CR2,NI,NI2 :  !

 PHASE LIQUID:L %  1  1.0  !
 CONST LIQUID:L :AL,CR,NI :  !

 PHASE FCC_A1  %E  2 1   1 !
 CONST FCC_A1  :AL,CR,NI% : VA% :  !

 PHASE BCC_A2  %F  2 1   3 !
 CONST BCC_A2  :AL,CR%,NI,VA : VA :  !

 PHASE HCP_A3  %A  2 1   .5 !
 CONST HCP_A3  :AL,CR,NI : VA% :  !

 PHASE BCC_B2  %BC  3 .5 .5    3 !
 CONST BCC_B2  :AL,CR,NI%,VA : AL%,CR,NI,VA : VA: !

 PHASE FCC_L12  %AD  3 .75   .25   1 !
 CONST FCC_L12  :AL,CR,NI : AL,CR,NI : VA :  !


 PHASE C14_LAVES % 2 2.0 1.0 !
 CONST C14_LAVES : AL,CR%,NI : AL,CR,NI : !

 PHASE C15_LAVES % 2 2.0 1.0 !
 CONST C15_LAVES : AL,CR%,NI : AL,CR,NI : !

 PHASE C36_LAVES % 2 2.0 1.0 !
 CONST C36_LAVES : AL,CR%,NI : AL,CR,NI : !

 PHASE SIGMA  %  3 8   4   18 !
 CONSTITUENT SIGMA  :AL,NI : CR : AL,CR%,NI :  !

 PHASE NEWSIGMA  %  3 10   4   16 !
 CONSTITUENT NEWSIGMA  :AL,NI : CR : AL,CR,NI :  !

 PHASE CHI_A12  %  3 24   10   24 !
 CONST CHI_A12  :CR,NI : CR : CR,NI :  !

 PHASE MTI2 % 2 1.0 2.0 !
 CONST MTI2 : CR,NI% : AL,CR,NI : !


 FUNCTION ZERO       298.15  0;,,N !
 FUNCTION DP         298.15  +P-101325;,,N !
 FUNCTION TROIS 298.15 3;,,N !
 FUNCTION UNTIER 298.15 TROIS**(-1);,,N !

$****************************************************************************
$
$                                                            UNARY PARAMETERS
$
$----------------------------------------------------------------------------
$
$                                                                          Al
$
$                                                                   FUNCTIONS
$
 FUNCTION F154T      298.15
    +323947.58-25.1480943*T-20.859*T*LN(T)
    +4.5665E-05*T**2-3.942E-09*T**3-24275.5*T**(-1);
                    4300.0  Y
    +342017.233-54.0526109*T-17.7891*T*LN(T)+6.822E-05*T**2
    -1.91111667E-08*T**3-14782200*T**(-1);
                    8200.0  Y
    +542396.07-411.214335*T+22.2419*T*LN(T)-.00349619*T**2
    +4.0491E-08*T**3-2.0366965E+08*T**(-1);  1.00000E+04  N !
$
 FUNCTION F625T      298.15
    +496408.232+35.479739*T-41.6397*T*LN(T)
    +.00249636*T**2-4.90507333E-07*T**3+85390.3*T**(-1);
                     900.00  Y
    +497613.221+17.368131*T-38.85476*T*LN(T)-2.249805E-04*T**2
    -9.49003167E-09*T**3-5287.23*T**(-1);  2.80000E+03  N !
$
 FUNCTION GHSERAL    298.15
    -7976.15+137.093038*T-24.3671976*T*LN(T)
    -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);
                     700.00  Y
    -11276.24+223.048446*T-38.5844296*T*LN(T)
    +.018531982*T**2-5.764227E-06*T**3+74092*T**(-1);
                     933.60  Y
    -11278.378+188.684153*T-31.748192*T*LN(T)
    -1.231E+28*T**(-9);,,  N !
$
 FUNCTION GHCPAL     298.15  +5481-1.8*T+GHSERAL;,,N !
$
 FUNCTION GBCCAL     298.15  +10083-4.813*T+GHSERAL;,,N !
$
 FUNCTION GLIQAL     298.14
    +11005.029-11.841867*T+7.934E-20*T**7+GHSERAL;
                     933.59  Y
    +10482.282-11.253974*T+1.231E+28*T**(-9)+GHSERAL;,,N !
$
$                                                                   GAS PHASE
$
 PARAMETER G(GAS,AL;0)  298.15  +F154T+R*T*LN(1E-05*P);,,N REF184 !
 PARAMETER G(GAS,AL2;0)  298.15  +F625T+R*T*LN(1E-05*P);,,N REF448 !
$
$                                                                LIQUID PHASE
$
 PARAMETER   G(LIQUID,AL;0)   298.13
      +11005.029-11.841867*T+7.934E-20*T**7+GHSERAL;
                                933.60  Y
      +10482.382-11.253974*T+1.231E+28*T**(-9)
      +GHSERAL;,,N 91DIN !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER G(FCC_A1,AL:VA;0)  298.15  +GHSERAL;,,N 91DIN !
$
$                                                                BCC_A2 PHASE
$
 PARAMETER G(BCC_A2,AL:VA;0)  298.15  +GBCCAL;,,N 91DIN !
   FUNC B2ALVA 295.15 10000-T;,,N !
   FUNC LB2ALVA 298.15 150000;,,N !
 PARAMETER L(BCC_A2,AL,VA:VA;0)  298.15  B2ALVA+LB2ALVA;,,N 99DUP !
$
$                                                                HCP_A3 PHASE
$
 PARAMETER G(HCP_A3,AL:VA;0)  298.15  +GHCPAL;,,N 91DIN !
$
$                                                                BCC_B2 PHASE
$
 PARAMETER G(BCC_B2,AL:VA:VA;0)  298.15  .5*B2ALVA-.5*LB2ALVA;,,N 99DUP !
 PARAMETER G(BCC_B2,VA:AL:VA;0)  298.15  .5*B2ALVA-.5*LB2ALVA;,,N 99DUP !
$
$                                                              ALTI_L10 PHASE
$
 PARAMETER G(ALTI_L10,AL:AL;0) 298.15 2*GHSERAL+4;,,N COST507 !
$
$                                                            ALTI3_DO19 PHASE
$
 PARAMETER G(ALTI3_DO19,AL:AL;0) 298.15 +4*GHCPAL;,,N COST507 !
$
$                                                             C14_LAVES PHASE
$
 PARAMETER G(C14_LAVES,AL:AL;0) 298.15 +3*GHSERAL+15000;,,N 95DUP8 !
$
$                                                             C15_LAVES PHASE
$
 PARAMETER G(C15_LAVES,AL:AL;0) 298.15 +3*GHSERAL+15000;,,N REFLAV !
$
$                                                             C36_LAVES PHASE
$
 PARAMETER G(C36_LAVES,AL:AL;0) 298.15 +3*GHSERAL+15000;,,N REFLAV !
$
$                                                                 H_L21 PHASE
$
 PARAMETER G(H_L21,AL:AL:VA;0) 298.15 +10000-T+GBCCAL;,,N 95DUP8 !
$
$                                                             NI3TI_DO24 PHASE
$
 PARAMETER G(NI3TI_DO24,AL:AL;0) 298.15 GHCPAL;,,N 95DUP8 !
$
$----------------------------------------------------------------------------
$
$                                                                          Cr
$
$                                                                   FUNCTIONS
$
 FUNCTION F7454T     298.15
    +390765.331-31.5192154*T-21.36083*T*LN(T)
    +7.253215E-04*T**2-1.588679E-07*T**3+10285.15*T**(-1);
                     1100.0  Y
    +393886.928-44.107465*T-19.96003*T*LN(T)+.001513089*T**2
    -4.23648333E-07*T**3-722515*T**(-1);
                     2000.0  Y
    +421372.003-231.888524*T+5.362886*T*LN(T)-.00848877*T**2
    +2.984635E-07*T**3-6015405*T**(-1);
                     3300.0  Y
    +305164.698+251.019831*T-55.20304*T*LN(T)+.005324585*T**2
    -2.850405E-07*T**3+34951485*T**(-1);
                     5100.0  Y
    +1069921.1-1708.93262*T+175.0508*T*LN(T)-.025574185*T**2
    +4.94447E-07*T**3-4.4276355E+08*T**(-1);
                     7600.0  Y
    -871952.838+1686.47356*T-204.5589*T*LN(T)+.007475225*T**2
    -4.618745E-08*T**3+1.423504E+09*T**(-1);  1.00000E+04  N !
$
 FUNCTION F7735T     298.15  +598511.402+41.5353219*T-40.56798*T*LN(T)
    +.004961847*T**2-1.61216717E-06*T**3+154422.85*T**(-1);
                     800.00  Y
    +613345.232-104.20799*T-19.7643*T*LN(T)-.007085085*T**2
    -4.69883E-07*T**3-1738066.5*T**(-1);
                     1400.0  Y
    +642608.843-369.286259*T+17.64743*T*LN(T)-.02767321*T**2
    +1.605906E-06*T**3-5831655*T**(-1);
                     2300.0  Y
    +553119.895+159.188556*T-52.07969*T*LN(T)-.004229401*T**2
    +1.5939925E-07*T**3+14793625*T**(-1);
                     3900.0  Y
    +347492.339+623.137624*T-105.0428*T*LN(T)+3.9699545E-04*T**2
    +1.51783483E-07*T**3+1.4843765E+08*T**(-1);
                     5800.0  Y
    -484185.055+2598.25559*T-334.7145*T*LN(T)+.028597625*T**2
    -4.97520167E-07*T**3+7.135805E+08*T**(-1);  6.00000E+03  N !
$
 FUNCTION GHSERCR    298.14
    -8856.94+157.48*T-26.908*T*LN(T)
    +.00189435*T**2-1.47721E-06*T**3+139250*T**(-1);
                     2180.0  Y
    -34869.344+344.18*T-50*T*LN(T)-2.88526E+32*T**(-9);,,N !
$
 FUNCTION GCRLIQ     298.15
    +24339.955-11.420225*T+2.37615E-21*T**7+GHSERCR;
                     2180.0  Y
    -16459.984+335.616316*T-50*T*LN(T);,,N !
$
 FUNCTION GFCCCR     298.15  +7284+.163*T+GHSERCR;,,N !
$
 FUNCTION GHCPCR     298.15  +4438+GHSERCR;,,N !
$
 FUNCTION ACRBCC     298.15  +1.7E-05*T+9.2E-09*T**2;,,N !
 FUNCTION BCRBCC     298.15  +1+2.6E-11*P;,,N !
 FUNCTION CCRBCC     298.15  2.08E-11;,,N !
 FUNCTION DCRBCC     298.15  +1*LN(BCRBCC);,,N !
 FUNCTION VCRBCC     298.15  +7.188E-06*EXP(ACRBCC);,,N !
 FUNCTION ECRBCC     298.15  +1*LN(CCRBCC);,,N !
 FUNCTION XCRBCC     298.15  +1*EXP(.8*DCRBCC)-1;,,N !
 FUNCTION YCRBCC     298.15  +VCRBCC*EXP(-ECRBCC);,,N !
 FUNCTION ZCRBCC     298.15  +1*LN(XCRBCC);,,N !
 FUNCTION GPCRBCC    298.15  +YCRBCC*EXP(ZCRBCC);,,N !
$
 FUNCTION ACRLIQ     298.15  +1.7E-05*T+9.2E-09*T**2;,,N !
 FUNCTION BCRLIQ     298.15  +1+4.65E-11*P;,,N !
 FUNCTION CCRLIQ     298.15  3.72E-11;,,N !
 FUNCTION DCRLIQ     298.15  +1*LN(BCRLIQ);,,N !
 FUNCTION VCRLIQ     298.15  +7.653E-06*EXP(ACRLIQ);,,N !
 FUNCTION ECRLIQ     298.15  +1*LN(CCRLIQ);,,N !
 FUNCTION XCRLIQ     298.15  +1*EXP(.8*DCRLIQ)-1;,,N !
 FUNCTION YCRLIQ     298.15  +VCRLIQ*EXP(-ECRLIQ);,,N !
 FUNCTION ZCRLIQ     298.15  +1*LN(XCRLIQ);,,N !
 FUNCTION GPCRLIQ    298.15  +YCRLIQ*EXP(ZCRLIQ);,,N !
$
$                                                                   GAS PHASE
$
 PARAMETER G(GAS,CR;0)  298.15  +F7454T+R*T*LN(1E-05*P);,,N REF4465 !
 PARAMETER G(GAS,CR2;0)  298.15  +F7735T+R*T*LN(1E-05*P);,,  N REF4591 !
$
$                                                                LIQUID PHASE
$
 PARAMETER G(LIQUID,CR;0)  298.15  +GCRLIQ+GPCRLIQ;,,  N 91DIN !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER G(FCC_A1,CR:VA;0)  298.15  +GFCCCR+GPCRBCC;,,N 89DIN !
 PARAMETER TC(FCC_A1,CR:VA;0)  298.15  -1109;,,N 89DIN !
 PARAMETER BMAGN(FCC_A1,CR:VA;0)  298.15  -2.46;,,N 89DIN !
$
$                                                                BCC_A2 PHASE
$
 PARAMETER G(BCC_A2,CR:VA;0)  298.15  +GHSERCR+GPCRBCC;,,N 91DIN !
 PARAMETER TC(BCC_A2,CR:VA;0)  298.15  -311.5;,,N 89DIN ! 
 PARAMETER BMAGN(BCC_A2,CR:VA;0)  298.15  -.008;,,N 89DIN !
 PARAMETER L(BCC_A2,CR,VA:VA;0)  298.15  100000;,,N 99DUP6 !   
$
$                                                                HCP_A3 PHASE
$
 PARAMETER G(HCP_A3,CR:VA;0)  298.15  +GHCPCR;,,N 91DIN !
 PARAMETER TC(HCP_A3,CR:VA;0)  298.15  -1109;,,N 89DIN !
 PARAMETER BMAGN(HCP_A3,CR:VA;0)  298.15  -2.46;,,N 89DIN !
$
$                                                                BCC_B2 PHASE
$
 PARAMETER G(BCC_B2,CR:VA:VA;0)  298.15  0;,,N 99DUP6 !
 PARAMETER G(BCC_B2,VA:CR:VA;0)  298.15  0;,,N 99DUP6 !
$
$                                                               CHI_A12 PHASE
$
 PARAMETER G(CHI_A12,CR:CR:CR;0)  298.15
     +48*GFCCCR+10*GHSERCR+109000+123*T;,,N 87GUS !
$
$                                                            ALTI3_DO19 PHASE
$
 PARAMETER G(ALTI3_DO19,CR:CR;0) 298.15 +4*GHCPCR;,,N COST507 !
$
$                                                             C14_LAVES PHASE
$
 PARAMETER G(C14_LAVES,CR:CR;0) 298.15 +3*GHSERCR+15000;,,N REFLAV !
$
$                                                             C15_LAVES PHASE
$
 PARAMETER G(C15_LAVES,CR:CR;0) 298.15 +3*GHSERCR+15000;,,N REFLAV !
$
$                                                             C36_LAVES PHASE
$
 PARAMETER G(C36_LAVES,CR:CR;0) 298.15 +3*GHSERCR+15000;,,N REFLAV !
$
$                                                                 MTI2 PHASE
$
 PARAMETER G(MTI2,CR:CR;0)  298.15 3*GHSERCR+15000;,,N 99DUP9  !
$
$                                                             NI3TI_DO24 PHASE
$
 PARAMETER G(NI3TI_DO24,CR:CR;0) 298.15 GHCPCR;,,N 95DUP9 !
$
$                                                              ALTI_L10 PHASE
$
 PARAMETER G(ALTI_L10,CR:CR;0) 298.15 2*GFCCCR;,,N COST507 !
$
$----------------------------------------------------------------------------
$
$                                                                          Ni
$
$                                                                   FUNCTIONS
$
 FUNCTION F13191T    298.15
    +417658.868-44.7777921*T-20.056*T*LN(T)
    -.0060415*T**2+1.24774E-06*T**3-16320*T**(-1);
                     800.00  Y
    +413885.448+9.41787679*T-28.332*T*LN(T)+.00173115*T**2
    -8.399E-08*T**3+289050*T**(-1);
                     3900.0  Y
    +440866.732-62.5810038*T-19.819*T*LN(T)+5.067E-04*T**2
    -4.93233333E-08*T**3-15879735*T**(-1);
                     7600.0  Y
    +848806.287-813.398164*T+64.69*T*LN(T)-.00731865*T**2
    +8.71833333E-08*T**3-3.875846E+08*T**(-1);  10000.  N !
$
 FUNCTION F13265T    298.15
    +638073.279-68.1901928*T-24.897*T*LN(T)
    -.0313584*T**2+5.93355333E-06*T**3-14215*T**(-1);
                     800.00  Y
    +611401.772+268.084821*T-75.25401*T*LN(T)+.01088525*T**2
    -7.08741667E-07*T**3+2633835*T**(-1);
                     2100.0  Y
    +637459.339+72.0712678*T-48.587*T*LN(T)-9.09E-05*T**2
    +9.12933333E-08*T**3-1191755*T**(-1);
                     4500.0 Y
    +564540.781+329.599011*T-80.11301*T*LN(T)+.00578085*T**2
    -1.08841667E-07*T**3+29137900*T**(-1);  6000.0  N !
$
 FUNCTION GHSERNI    298.14
    -5179.159+117.854*T-22.096*T*LN(T)
    -.0048407*T**2;
                     1728.0  Y
    -27840.655+279.135*T-43.1*T*LN(T)+1.12754E+31*T**(-9);,,  N   !
$
 FUNCTION GHCPNI     298.15  +1046+1.2552*T+GHSERNI;,,N !
$
 FUNCTION GBCCNI     298.15  +8715.084-3.556*T+GHSERNI;,,,   N !
$
$                                                                   GAS PHASE
$
 PARAMETER G(GAS,NI;0)  298.15  +F13191T+R*T*LN(1E-05*P);,,N REF7504 !
 PARAMETER G(GAS,NI2;0)  298.15 +F13265T+R*T*LN(1E-05*P);,,N REF7553 !
$
$                                                                LIQUID PHASE
$
 PARAMETER G(LIQUID,NI;0) 298.13
      +16414.686-9.397*T-3.82318E-21*T**7+GHSERNI;
                            1728.0  Y
      +18290.88-10.537*T-1.12754E+31*T**(-9)
      +GHSERNI;,,N 91DIN !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER G(FCC_A1,NI:VA;0)  298.15  +GHSERNI;,,N 91DIN !
 PARAMETER TC(FCC_A1,NI:VA;0)  298.15  633;,,N 89DIN !
 PARAMETER BMAGN(FCC_A1,NI:VA;0)  298.15  .52;,,N 89DIN !
$
$                                                                BCC_A2 PHASE
$
 PARAMETER G(BCC_A2,NI:VA;0)  298.15  +GBCCNI;,,N 91DIN !
 PARAMETER TC(BCC_A2,NI:VA;0)  298.15  575;,,N 89DIN !
 PARAMETER BMAGN(BCC_A2,NI:VA;0)  298.15  .85;,,N 89DIN !
   FUNC B2NIVA 295.15 +162397.3-27.40575*T;,,N !
   FUNC LB2NIVA 298.15 -64024.38+26.49419*T;,,N !
 PARAMETER L(BCC_A2,NI,VA:VA;0)  298.15  B2NIVA+LB2NIVA;,,N 99DUP !   
$
$                                                                HCP_A3 PHASE
$
 PARAMETER G(HCP_A3,NI:VA;0)  298.15  +GHCPNI;,,N 91DIN !
 PARAMETER TC(HCP_A3,NI:VA;0)  298.15  633;,,N 86FER1 !
 PARAMETER BMAGN(HCP_A3,NI:VA;0)  298.15  .52;,,N 86FER1 !
$
$                                                                BCC_B2 PHASE
$
 PARAMETER G(BCC_B2,VA:NI:VA;0)  298.15  .5*B2NIVA-.5*LB2NIVA;,,N 99DUP !
 PARAMETER G(BCC_B2,NI:VA:VA;0)  298.15  .5*B2NIVA-.5*LB2NIVA;,,N 99DUP !
$
$                                                             NI3TI_DO24 PHASE
$
 PARAMETER G(NI3TI_DO24,NI:NI;0) 298.15 +GHCPNI;,,N 90SAU !
$
$                                                            ALTI3_DO19 PHASE
$
 PARAMETER G(ALTI3_DO19,NI:NI;0) 298.15 +4*GHCPNI;,,N COST507 !
$
$                                                             C14_LAVES PHASE
$
 PARAMETER G(C14_LAVES,NI:NI;0) 298.15 +15000+3*GHSERNI;,,N REFLAV !
$
$                                                             C15_LAVES PHASE
$
 PARAMETER G(C15_LAVES,NI:NI;0) 298.15 +15000+3*GHSERNI;,,N REFLAV !
$
$                                                             C36_LAVES PHASE
$
 PARAMETER G(C36_LAVES,NI:NI;0) 298.15 +15000+3*GHSERNI;,,N REFLAV !
$
$                                                                 H_L21 PHASE
$
 PARAMETER G(H_L21,NI:NI:NI;0) 298.15 +2*GBCCNI;,,N 95DUP8 !
 PARAMETER G(H_L21,NI:NI:VA;0) 298.15 +100000+GHSERNI;,,N 95DUP8 !
$
$                                                                 MTI2 PHASE
$
 PARAMETER G(MTI2,NI:NI;0)  298.15 3*GHSERNI+15000;,,N 99DUP9  !
$
$****************************************************************************
$
$                                                           BINARY PARAMETERS
$
$----------------------------------------------------------------------------
$
$                                                                       Al-Cr
$                             Mainly from Saunders (COST507)
$                             Metastable B2 and L12 from revision of Al-Cr-Ni
$
$                                                                LIQUID PHASE
$
 PARAMETER L(LIQUID,AL,CR;0)  298.15  -29000;,,N 91SAU1 !
 PARAMETER L(LIQUID,AL,CR;1)  298.15  -11000;,,N 91SAU1 !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER G(FCC_A1,AL,CR:VA;0)  298.15  -45900+6*T;,,N 91SAU1 !
$
$                                                                BCC_A2 PHASE
$
 PARAMETER G(BCC_A2,AL,CR:VA;0)  298.15  -54900+10*T;,,N 91SAU1 ! 
$
$                                                                BCC_B2 PHASE
$                                                                  metastable
$ Present work: july 1999, study of Al-Cr-Ni, revision of NDTH. The B2
$ phase is not stabilized enough to become stable in the Al-Cr. It is
$ thus not in agreement with "T. Helander, and O. Tolochko, J. of Phase
$ Eq, 20 (1) 1999, 57-60." Further study on the extension of the B2 phase
$ towards AlCr in Al-Cr-Ni would be desirable.
 PARAMETER G(BCC_B2,AL:CR:VA;0)  298.15  -2000;,,N 99DUP6  !
 PARAMETER G(BCC_B2,CR:AL:VA;0)  298.15  -2000;,,N 99DUP6 !
$
$                                                               FCC_L12 PHASE
$                                                                  metastable
$ Present work: july 1999, study of Al-Cr-Ni, revision of NDTH.
     FUN U1ALCR  298.15 -830;,,N  99DUP6 !
     FUN U3ALCR  298.15 0.0; 6000.00   99DUP6 !
     FUN U4ALCR  298.15 0.0; 6000.00 N  99DUP6 !
   FUNCTION L04ALCR 298.15 U3ALCR;,,N !
   FUNCTION L14ALCR 298.15 U4ALCR;,,N !
   FUNCTION ALCR3 298.15 3*U1ALCR;,,N !
   FUNCTION AL2CR2 298.15 4*U1ALCR;,,N !
   FUNCTION AL3CR 298.15 3*U1ALCR;,,N !
 PARAMETER G(FCC_L12,CR:AL:VA;0)  298.15  +ALCR3;,,  N  99DUP6 !
 PARAMETER G(FCC_L12,AL:CR:VA;0)  298.15  +AL3CR;,,  N  99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:AL:VA;0) 298.15
     -1.5*ALCR3+1.5*AL2CR2+1.5*AL3CR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:CR:VA;0) 298.15
     +1.5*ALCR3+1.5*AL2CR2-1.5*AL3CR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:AL:VA;1) 298.15
     +0.5*ALCR3-1.5*AL2CR2+1.5*AL3CR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:CR:VA;1) 298.15
     -1.5*ALCR3+1.5*AL2CR2-0.5*AL3CR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,*:AL,CR:VA;0) 298.15 +L04ALCR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,*:AL,CR:VA;1) 298.15 +L14ALCR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:*:VA;0) 298.15 +3*L04ALCR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:*:VA;1) 298.15 +3*L14ALCR;,,N 99DUP6 !
$
$                                                               AL11CR2 PHASE
$
 PHASE AL11CR2  %  3 10   1   2 !
 CONST AL11CR2  :AL : AL : CR : !
 PARAMETER G(AL11CR2,AL:AL:CR;0)  298.15
     +11*GHSERAL+2*GHSERCR-175500+25.805*T;,,N 91SAU1 !
$
$                                                               AL13CR2 PHASE
$
 PHASE AL13CR2  %  2 13   2 !
 CONST AL13CR2  :AL : CR :  !
 PARAMETER G(AL13CR2,AL:CR;0)  298.15
     +13*GHSERAL+2*GHSERCR-174405+22.2*T;,,N 91SAU1 !
$
$                                                                 AL4CR PHASE
$
 PHASE AL4CR  %  2 4   1 !
 CONST AL4CR  :AL : CR :  !
 PARAMETER G(AL4CR,AL:CR;0)  298.15
     +4*GHSERAL+GHSERCR-89025+19.05*T;,,N 91SAU1 !
$
$                                                              AL8CR5_H PHASE
$
 PHASE AL8CR5_H  %  2 8   5 !
 CONST AL8CR5_H  :AL : CR :  !
 PARAMETER G(AL8CR5_H,AL:CR;0)  298.15
     +8*GHSERAL+5*GHSERCR-147732-58.5*T;,,N 91SAU1 !
$
$                                                              AL8CR5_L PHASE
$
 PHASE AL8CR5_L  %  2 8   5 !
 CONST AL8CR5_L  :AL : CR :  !
 PARAMETER G(AL8CR5_L,AL:CR;0)  298.15
     +8*GHSERAL+5*GHSERCR-229515;,,N 91SAU1 !
$
$                                                              AL9CR4_H PHASE
$
 PHASE AL9CR4_H  %  2 9   4 !
 CONST AL9CR4_H  :AL : CR :  !
 PARAMETER G(AL9CR4_H,AL:CR;0)  298.15
     +9*GHSERAL+4*GHSERCR-134433-56.16*T;,,N 91SAU1 !
$
$                                                              AL9CR4_L PHASE
$
 PHASE AL9CR4_L  %  2 9   4 !
 CONST AL9CR4_L  :AL : CR :  !
 PARAMETER G(AL9CR4_L,AL:CR;0)  298.15
     +9*GHSERAL+4*GHSERCR-230750+16.094*T;,,N 91SAU1 !
$
$                                                                 ALCR2 PHASE
$
 PHASE ALCR2  %  2 1   2 !
 CONST ALCR2  :AL : CR :  !
 PARAMETER G(ALCR2,AL:CR;0)  298.15
     +GHSERAL+2*GHSERCR-32700-8.79*T;,,N 91SAU1 !
$
$                                                            AL3TI_DO22 PHASE
$                                                                  metastable
$
 PARAMETER G(AL3TI_DO22,AL:CR;0)  298.15
     +3*GHSERAL+GHSERCR-53000;,,N COST507 !
$
$                                                            NI3TI_DO24 PHASE
$                                                                  metastable
$
 PARA G(NI3TI_DO24,CR:AL;0)  298.15  .25*GHCPAL+.75*GHCPCR;,,N LINEAR  !
 PARA G(NI3TI_DO24,AL:CR;0)  298.15  .75*GHCPAL+.25*GHCPCR;,,N LINEAR  !
$
$                                                                 MTI2 PHASE
$                                                                  metastable
$
 PARA G(MTI2,CR:AL;0)  298.15  +GHSERCR+2*GHSERAL;,,N LINEAR  !
$
$                                                            ALTI3_DO19 PHASE
$                                                                  metastable
$
 PARA G(ALTI3_DO19,CR:AL;0)  298.15  +GHCPAL+3*GHCPCR;,,N COST507 !
 PARA G(ALTI3_DO19,AL:CR;0)  298.15  +3*GHCPAL+GHCPCR;,,N COST507 !
 PARA L(ALTI3_DO19,AL,CR:AL;0)  298.15  1E-04;,,N  COST507 !
 PARA L(ALTI3_DO19,AL:AL,CR;0)  298.15  1E-04;,,N COST507 !
 PARA L(ALTI3_DO19,CR:AL,CR;0)  298.15  1E-04;,,N COST507 !
 PARA L(ALTI3_DO19,AL,CR:CR;0)  298.15  1E-04;,,N COST507 !
$
$                                                              ALTI_L10 PHASE
$                                                                  metastable
$
 PARA G(ALTI_L10,CR:AL;0)  298.15  -25000+3*T+GHSERAL+GFCCCR;,,N COST507 !
 PARA G(ALTI_L10,AL:CR;0)  298.15  -25000+3*T+GHSERAL+GFCCCR;,,N COST507 !
 PARA L(ALTI_L10,AL,CR:AL;0)  298.15  1E-04;,,N COST507 !
 PARA L(ALTI_L10,AL:AL,CR;0)  298.15  1E-04;,,N COST507 !
 PARA L(ALTI_L10,CR:AL,CR;0)  298.15  1E-04;,,N COST507 !
 PARA L(ALTI_L10,AL,CR:CR;0)  298.15  1E-04;,,N COST507 !
$
$                                                                 SIGMA PHASE
$                                                                  metastable
$
 PARA G(SIGMA,AL:CR:AL;0)  298.15
   8*GHSERAL+4*GHSERCR+18*GBCCAL+UNASS;,,N 99LEE !
 PARA G(SIGMA,AL:CR:CR;0)  298.15
   8*GHSERAL+4*GHSERCR+18*GHSERCR+UNASS;,,N 99LEE !
$
$                                                              NEWSIGMA PHASE
$                                                                  metastable
$
 PARA G(NEWSIGMA,AL:CR:AL;0)  298.15
   +10*GHSERAL+4*GHSERCR+16*GBCCAL;,,N LINEAR !
 PARA G(NEWSIGMA,AL:CR:CR;0)  298.15
   +10*GHSERAL+4*GHSERCR+16*GHSERCR;,,N LINEAR !
$
$                                                             C14_LAVES PHASE
$                                                                  metastable
$
 PARAMETER G(C14_LAVES,CR:AL;0)  298.15
     +2*GHSERCR+GHSERAL;,,N LINEAR !
 PARAMETER G(C14_LAVES,AL:CR;0) 298.15 
     +GHSERCR+2*GHSERAL;,,N LINEAR !
$
$                                                             C15_LAVES PHASE
$                                                                  metastable
$
 PARAMETER G(C15_LAVES,CR:AL;0)  298.15
     +2*GHSERCR+GHSERAL;,,N LINEAR !
 PARAMETER G(C15_LAVES,AL:CR;0) 298.15 
     +GHSERCR+2*GHSERAL;,,N LINEAR !
$
$                                                             C36_LAVES PHASE
$                                                                  metastable
$
 PARAMETER G(C36_LAVES,CR:AL;0)  298.15
     +2*GHSERCR+GHSERAL;,,N LINEAR !
 PARAMETER G(C36_LAVES,AL:CR;0) 298.15 
     +GHSERCR+2*GHSERAL;,,N LINEAR !
$
$----------------------------------------------------------------------------
$
$                                                                       Al-Ni
$                     Mainly from ND thesis,
$                     slighly revised to get better solvus at low temperature
$
$                                                                LIQUID PHASE
$
 PARAMETER L(LIQUID,AL,NI;0)  298.15 -207109.28+41.31501*T;,,N 95DUP3 !
 PARAMETER L(LIQUID,AL,NI;1)  298.15 -10185.79+5.8714*T;,,N 95DUP3 !
 PARAMETER L(LIQUID,AL,NI;2)  298.15 +81204.81-31.95713*T;,,N 95DUP3 !
 PARAMETER L(LIQUID,AL,NI;3)  298.15  +4365.35-2.51632*T;,,N 95DUP3 !
 PARAMETER L(LIQUID,AL,NI;4)  298.15  -22101.64+13.16341*T;,,N 95DUP3 !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER TC(FCC_A1,AL,NI:VA;0)  298.15  -1112;,,N 95DUP3 !
 PARAMETER TC(FCC_A1,AL,NI:VA;1)  298.15  1745;,,N 95DUP3 !
 PARAMETER G(FCC_A1,AL,NI:VA;0)  298.15  -162407.75+16.212965*T;,,N 95DUP3 !
 PARAMETER G(FCC_A1,AL,NI:VA;1)  298.15  +73417.798-34.914168*T;,,N 95DUP3 !   
 PARAMETER G(FCC_A1,AL,NI:VA;2)  298.15  +33471.014-9.8373558*T;,,N 95DUP3 !   
 PARAMETER G(FCC_A1,AL,NI:VA;3)  298.15  -30758.01+10.25267*T;,,N 95DUP3 !
$
$                                                                BCC_A2 PHASE
$                                                                  metastable
$
   FUNC B2ALNI 295.15 -152397.3+26.40575*T;,,N !
   FUNC LB2ALNI 298.15 -52440.88+11.30117*T;,,N !
 PARAMETER L(BCC_A2,AL,NI:VA;0)  298.15  B2ALNI+LB2ALNI;,,N 99DUP!   
$
$                                                                HCP_A3 PHASE
$                                                                  metastable
$
 PARAMETER TC(HCP_A3,AL,NI:VA;0)  298.15  -1112;,,N 95DUP8 !
 PARAMETER TC(HCP_A3,AL,NI:VA;1)  298.15  1745;,,N 95DUP8 !
 PARAMETER L(HCP_A3,AL,NI:VA;0)  298.15  -162407.75+16.212965*T;,,N 95DUP8 !
 PARAMETER L(HCP_A3,AL,NI:VA;1)  298.15  +73417.798-34.914168*T;,,N 95DUP8 !   
 PARAMETER L(HCP_A3,AL,NI:VA;2)  298.15  +33471.014-9.8373558*T;,,N 95DUP8 !   
 PARAMETER L(HCP_A3,AL,NI:VA;3)  298.15  -30758.01+10.25267*T;,,N 95DUP8 !
$
$                                                                BCC_B2 PHASE
$
 PARAMETER G(BCC_B2,AL:NI:VA;0)  298.15  .5*B2ALNI-.5*LB2ALNI;,,N 99DUP !
 PARAMETER G(BCC_B2,NI:AL:VA;0)  298.15  .5*B2ALNI-.5*LB2ALNI;,,N 99DUP !
$
$                                                               FCC_L12 PHASE
$
     FUN UALNI 298.15 -22212.8931+4.39570389*T;,,,N 99DUP3 !
     FUN U1ALNI 298.15 2*UNTIER*UALNI;,,,N 99DUP3 !
     FUN U3ALNI 298.15 0;,,,N 99DUP3 !
     FUN U4ALNI 298.15 7203.60609-3.74273030*T;,,,N 99DUP3 !
   FUNCTION L04ALNI 298.15 U3ALNI;,,N 99DUP3 !
   FUNCTION L14ALNI 298.15 U4ALNI;,,N 99DUP3 !
   FUNCTION ALNI3   298.15 +3*U1ALNI;,,,N 99DUP3 !
   FUNCTION AL2NI2  298.15 +4*U1ALNI;,,,N 99DUP3 !
   FUNCTION AL3NI   298.15 +3*U1ALNI;,,,N 99DUP3 !
 PARAMETER G(FCC_L12,NI:AL:VA;0)  298.15  +ALNI3;,,N 99DUP3 !
 PARAMETER G(FCC_L12,AL:NI:VA;0)  298.15  +AL3NI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:AL:VA;0) 298.15
     -1.5*ALNI3+1.5*AL2NI2+1.5*AL3NI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:NI:VA;0) 298.15
     +1.5*ALNI3+1.5*AL2NI2-1.5*AL3NI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:AL:VA;1) 298.15
     +0.5*ALNI3-1.5*AL2NI2+1.5*AL3NI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:NI:VA;1) 298.15
     -1.5*ALNI3+1.5*AL2NI2-0.5*AL3NI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,*:AL,NI:VA;0) 298.15 +L04ALNI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,*:AL,NI:VA;1) 298.15 +L14ALNI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:*:VA;0) 298.15 +3*L04ALNI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:*:VA;1) 298.15 +3*L14ALNI;,,N 99DUP3 !
$
$                                                                AL3NI1 PHASE
$
 PHASE AL3NI1  %  2 .75   .25 !
 CONST AL3NI1  :AL : NI :  !
 PARAMETER G(AL3NI1,AL:NI;0)  298.15
 -48483.73+12.29913*T
 +.75*GHSERAL+.25*GHSERNI;,,N 95DUP3 !
$
$                                                                AL3NI2 PHASE
$
 PHASE AL3NI2  %  3 3   2   1 !
 CONST AL3NI2  :AL : AL,NI% : NI,VA% :  !
 PARAMETER  G(AL3NI2,AL:AL:NI;0)  298.15  +5*GBCCAL+GBCCNI
     -39465.978+7.89525*T;,,N 95DUP3 !
 PARAMETER G(AL3NI2,AL:NI:NI;0)  298.15  +3*GBCCAL+3*GBCCNI
     -427191.9+79.21725*T;,,N 95DUP3 !
 PARAMETER  G(AL3NI2,AL:AL:VA;0) 298.15  +5*GBCCAL
     +30000-3*T;,,N 95DUP3 !
 PARAMETER  G(AL3NI2,AL:NI:VA;0)  298.15  +3*GBCCAL+2*GBCCNI
     -357725.92+68.322*T;,,N 95DUP3 !
 PARAMETER  L(AL3NI2,AL:AL,NI:*;0)  298.15
     -193484.18+131.79*T;,,N 95DUP3 !
 PARAMETER  L(AL3NI2,AL:*:NI,VA;0)  298.15
     -22001.7+7.0332*T;,,N 95DUP3 !
$
$                                                                AL3NI5 PHASE
$
 PHASE AL3NI5  %  2 .375   .625 !
 CONST AL3NI5  :AL : NI :  !
 PARAMETER G(AL3NI5,AL:NI;0)  298.15  +.375*GHSERAL+.625*GHSERNI
     -55507.7594+7.2648103*T;,,N 95DUP3 !
$
$                                                                 MTI2 PHASE
$                                                                  metastable
$
 PARAMETER G(MTI2,NI:AL;0) 298.15 GHSERNI+2*GHSERAL+80810;,,N 95DUP8 !
$
$                                                             C14_LAVES PHASE
$                                                                  metastable
$
 PARAMETER G(C14_LAVES,NI:AL;0)  298.15
     +2*GHSERNI+GHSERAL+130000;,,N 99DUP8 !
 PARAMETER G(C14_LAVES,AL:NI;0) 298.15 
     +GHSERNI+2*GHSERAL-100000;,,N 99DUP8 !
$
$                                                             C15_LAVES PHASE
$                                                                  metastable
$
 PARAMETER G(C15_LAVES,NI:AL;0)  298.15
     +2*GHSERNI+GHSERAL+130000-2000;,,N 99DUP8 !
 PARAMETER G(C15_LAVES,AL:NI;0) 298.15 
     +GHSERNI+2*GHSERAL-100000+2000;,,N 99DUP8 !
$
$                                                             C36_LAVES PHASE
$                                                                  metastable
$
 PARAMETER G(C36_LAVES,NI:AL;0)  298.15
     +2*GHSERNI+GHSERAL+130000-3000;,,N 99DUP8 !
 PARAMETER G(C36_LAVES,AL:NI;0) 298.15 
     +GHSERNI+2*GHSERAL-100000+3000;,,N 99DUP8 !
$
$                                                                 H_L21 PHASE
$                                                                  metastable
$
  FUNCTION GHALNI 298.15  -152397.3+26.40575*T+GBCCAL+GBCCNI+4000;,,N  99DUP8 !
 PARAMETER G(H_L21,AL:AL:NI;0)  298.15  +GHALNI;,,N 99DUP8 !
 PARAMETER G(H_L21,NI:AL:NI;0)  298.15  +.5*GHALNI+GBCCNI;,,N 99DUP8 !
 PARAMETER G(H_L21,AL:NI:NI;0)  298.15  +.5*GHALNI+GBCCNI;,,N 99DUP8 !
 PARAMETER G(H_L21,NI:AL:VA;0)  298.15  +.5*GBCCAL+.5*GHSERNI
     +55000-.5*T;,,N 99DUP8 !
 PARAMETER G(H_L21,AL:NI:VA;0)  298.15  +.5*GBCCAL+.5*GHSERNI
     +55000-.5*T;,,N 99DUP8 !
$
$                                                             NI3TI_DO24 PHASE
$                                                                  metastable
$
  FUN DO24NIAL 298.15 +3*GHCPNI+GHCPAL-80000;,,N 99DUP8 !
 PARAMETER G(NI3TI_DO24,NI:AL;0)  298.15  
     .25*DO24NIAL;,,N 99DUP8 !
 PARAMETER G(NI3TI_DO24,AL:NI;0) 298.15
     +.75*GHCPAL+.25*GHCPNI;,,N 95DUP8 !
$
$                                                             ALTI3_DO19 PHASE
$                                                                  metastable
$
 PARAMETER G(ALTI3_DO19,NI:AL;0)  298.15  
     3*GHCPNI+GHCPAL;,,N 99DUP8 !
 PARAMETER G(ALTI3_DO19,AL:NI;0) 298.15
     3*GHCPAL+GHCPNI;,,N 95DUP8 !
$
$----------------------------------------------------------------------------
$
$                                                                       Cr-Ni
$                             Mainly from SSOL
$                             Metastable B2 and L12 from revision of Al-Cr-Ni
$ 
$                                                                LIQUID PHASE
$
 PARAMETER L(LIQUID,CR,NI;0)  298.15  +318-7.3318*T;,,N 91LEE !
 PARAMETER L(LIQUID,CR,NI;1)  298.15  +16941-6.3696*T;,,N 91LEE !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER G(FCC_A1,CR,NI:VA;0)  298.15  +8030-12.8801*T;,,N 91LEE !
 PARAMETER G(FCC_A1,CR,NI:VA;1)  298.15  +33080-16.0362*T;,,N 91LEE !   
 PARAMETER TC(FCC_A1,CR,NI:VA;0)  298.15  -3605;,,N 86DIN !
 PARAMETER BMAGN(FCC_A1,CR,NI:VA;0)  298.15  -1.91;,,N 86DIN !
$
$                                                                BCC_A2 PHASE
$
 PARAMETER G(BCC_A2,CR,NI:VA;0)  298.15  +17170-11.8199*T;,,N 91LEE !
 PARAMETER G(BCC_A2,CR,NI:VA;1)  298.15  +34418-11.8577*T;,,N 91LEE !
 PARAMETER TC(BCC_A2,CR,NI:VA;0)  298.15  2373;,,N 86DIN !
 PARAMETER TC(BCC_A2,CR,NI:VA;1)  298.15  617;,,N 86DIN !
 PARAMETER BMAGN(BCC_A2,CR,NI:VA;0)  298.15  4;,,N 86DIN !
$
$                                                                HCP_A3 PHASE
$                                                                  metastable
$
 PARAMETER G(HCP_A3,CR,NI:VA;0)  298.15  50000;,,N 95DUP6 !
 PARAMETER TC(HCP_A3,CR,NI:VA;0)  298.15  -3605;,,N 95DUP6 !
 PARAMETER BMAGN(HCP_A3,CR,NI:VA;0)  298.15  -1.91;,,N 95DUP6 !
$
$                                                                BCC_B2 PHASE
$                                                                  metastable
$
$ Present work: july 1999, study of Al-Cr-Ni, revision of NDTH.
 PARAMETER G(BCC_B2,CR:NI:VA;0)  298.15  4000;,,N 99DUP6 !
 PARAMETER G(BCC_B2,NI:CR:VA;0)  298.15  4000;,,N 99DUP6 !
$
$                                                               FCC_L12 PHASE
$                                                                  metastable
$ Present work: july 1999, study of Al-Cr-Ni, revision of NDTH.
$ The L12 phase is metastable in the binary Cr-Ni while it was stable in NDTH.
     FUN U1CRNI 298.15 -1980;,,,N 99DUP6 !
$     FUN U1CRNI 298.15 -7060+3.63*T;,,,N 99DUP6 !
     FUN U3CRNI 298.15 0;,,,N 99DUP6 !
     FUN U4CRNI 298.15 0;,,,N 99DUP6 !
   FUNCTION L04CRNI 298.15 U3CRNI;,,N 99DUP6 !
   FUNCTION L14CRNI 298.15 U4CRNI;,,N 99DUP6 !
   FUNCTION CRNI3   298.15 +3*U1CRNI;,,,N 99DUP6 !
   FUNCTION CR2NI2  298.15 +4*U1CRNI;,,,N 99DUP6 !
   FUNCTION CR3NI   298.15 +3*U1CRNI;,,,N 99DUP6 !
 PARAMETER G(FCC_L12,NI:CR:VA;0)  298.15  +CRNI3;,,  N 99DUP6 !
 PARAMETER G(FCC_L12,CR:NI:VA;0)  298.15  +CR3NI;,,  N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:CR:VA;0) 298.15
     -1.5*CRNI3+1.5*CR2NI2+1.5*CR3NI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:NI:VA;0) 298.15
     +1.5*CRNI3+1.5*CR2NI2-1.5*CR3NI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:CR:VA;1) 298.15
     +0.5*CRNI3-1.5*CR2NI2+1.5*CR3NI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:NI:VA;1) 298.15
     -1.5*CRNI3+1.5*CR2NI2-0.5*CR3NI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,*:CR,NI:VA;0) 298.15 +L04CRNI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,*:CR,NI:VA;1) 298.15 +L14CRNI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:*:VA;0) 298.15 +3*L04CRNI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:*:VA;1) 298.15 +3*L14CRNI;,,N 99DUP6 !
$                                                        
$                                                                 SIGMA PHASE
$                                                                  metastable
$
 PARAMETER G(SIGMA,NI:CR:CR;0)  298.15
   +8*GHSERNI+4*GHSERCR+18*GHSERCR+221157-227*T;,,N 91LEE !
 PARAMETER G(SIGMA,NI:CR:NI;0)  298.15
   +8*GHSERNI+4*GHSERCR+18*GBCCNI+175400;,,N 86GUS !
$                                                        
$                                                              NEWSIGMA PHASE
$                                                                  metastable
$
 PARAMETER G(NEWSIGMA,NI:CR:CR;0)  298.15
   +10*GHSERNI+4*GHSERCR+16*GHSERCR+221157-227*T;,,N SIG10416 !
 PARAMETER G(NEWSIGMA,NI:CR:NI;0)  298.15
   +10*GHSERNI+4*GHSERCR+16*GBCCNI+175400;,,N SIG10416 !
$                                                        
$                                                               CHI_A12 PHASE
$                                                                  metastable
$
 PARAMETER G(CHI_A12,NI:CR:CR;0)  298.15
    +24*GHSERNI+10*GHSERCR+24*GFCCCR;,,  N 99LEE !
 PARAMETER G(CHI_A12,CR:CR:NI;0)  298.15
    +24*GFCCCR+10*GHSERCR+24*GHSERNI;,,  N 99LEE !
 PARAMETER G(CHI_A12,NI:CR:NI;0)  298.15
    +24*GHSERNI+10*GHSERCR+24*GHSERNI;,, N 99LEE !
$
$                                                             C14_LAVES PHASE
$                                                                  metastable
$
 PARAMETER G(C14_LAVES,NI:CR;0) 298.15 2*GHSERNI+GHSERCR+15000;,,N 95DUP4 !
 PARAMETER G(C14_LAVES,CR:NI;0) 298.15 2*GHSERCR+GHSERNI+15000;,,N 95DUP4 !
$
$                                                             C15_LAVES PHASE
$                                                                  metastable
$
 PARAMETER G(C15_LAVES,NI:CR;0) 298.15 +2*GHSERNI+GHSERCR+15000;,,N 95DUP4 !
 PARAMETER G(C15_LAVES,CR:NI;0) 298.15 +2*GHSERNI+GHSERCR+15000;,,N 95DUP4 !
$
$                                                             C36_LAVES PHASE
$                                                                  metastable
$
 PARAMETER G(C36_LAVES,NI:CR;0) 298.15 +2*GHSERNI+GHSERCR+15000;,,N 99DUP9 !
 PARAMETER G(C36_LAVES,CR:NI;0) 298.15 +2*GHSERNI+GHSERCR+15000;,,N 99DUP9 !
$
$                                                                 MTI2 PHASE
$                                                                  metastable
$
 PARAMETER G(MTI2,NI:CR;0) 298.15 GHSERNI+2*GHSERCR+5000;,,N 95DUP9 !
 PARAMETER G(MTI2,CR:NI;0) 298.15 GHSERCR+2*GHSERNI+5000;,,N 95DUP9 !
$
$                                                             NI3TI_DO24 PHASE
$                                                                  metastable
$
 PARAMETER G(NI3TI_DO24,CR:NI;0) 298.15 .75*GHCPCR+.25*GHCPNI;,,N 95DUP9 !
 PARAMETER G(NI3TI_DO24,NI:CR;0) 298.15 .75*GHCPNI+.25*GHCPCR;,,N 95DUP9 !
$
$                                                             ALTI3_DO19 PHASE
$                                                                  metastable
$
 PARAMETER G(ALTI3_DO19,NI:CR;0)  298.15  
     3*GHCPNI+GHCPCR;,,N LINEAR !
 PARAMETER G(ALTI3_DO19,CR:NI;0) 298.15
     3*GHCPCR+GHCPNI;,,N LINEAR !
$
$****************************************************************************
$
$                                                          TERNARY PARAMETERS
$
$----------------------------------------------------------------------------
$
$                                                                    Al-Cr-Ni
$                                    July 1999, ND
$                                    Revision. Main changes:
$                                    - description of the A2/B2
$                                    - new liquidus data taken into account
$                                    - simpler ternary interaction parameters
$
$                                                                LIQUID PHASE
$
 PARAMETER L(LIQUID,AL,CR,NI;0)  298.15  16000;,,N 99DUP6 !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER G(FCC_A1,AL,CR,NI:VA;0)  298.15  30300;,,N 99DUP6 !
$
$                                                                BCC_A2 PHASE
$
 PARAMETER G(BCC_A2,AL,CR,NI:VA;0)  298.15  42500;,,N 99DUP6 !
$
$                                                               FCC_L12 PHASE
$
   FUN U1ALCRNI 298.15 6650;,,N 99DUP6 !
   FUN U2ALCRNI 298.15 0;,,N 99DUP6 !
   FUN U3ALCRNI 298.15 0;,,N 99DUP6 !
   FUN ALCRNI2 298.15 U1ALCR+2*U1ALNI+2*U1CRNI+U1ALCRNI;,,N 99DUP6 !
   FUN ALCR2NI 298.15 2*U1ALCR+U1ALNI+2*U1CRNI+U2ALCRNI;,,N 99DUP6 !
   FUN AL2CRNI 298.15 2*U1ALCR+2*U1ALNI+U1CRNI+U3ALCRNI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,CR,NI:AL:VA;0) 298.15
     -1.5*ALCRNI2-1.5*ALCR2NI+ALCR3+ALNI3+6*AL2CRNI
     -1.5*AL2CR2-1.5*AL2NI2-1.5*AL3CR-1.5*AL3NI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,CR,NI:CR:VA;0) 298.15
     -1.5*ALCRNI2+6*ALCR2NI-1.5*ALCR3-1.5*AL2CRNI
     -1.5*AL2CR2+AL3CR+CRNI3-1.5*CR2NI2-1.5*CR3NI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,CR,NI:NI:VA;0) 298.15
     +6*ALCRNI2-1.5*ALCR2NI-1.5*ALNI3-1.5*AL2CRNI
     -1.5*AL2NI2+AL3NI-1.5*CRNI3-1.5*CR2NI2+CR3NI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,CR:NI:VA;0) 298.15
     +1.5*ALCR2NI+1.5*AL2CRNI-1.5*AL3NI-1.5*CR3NI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,NI:CR:VA;0) 298.15
     +1.5*ALCRNI2+1.5*AL2CRNI-1.5*AL3CR-1.5*CRNI3;,,N 99DUP6 !
 PARA L(FCC_L12,CR,NI:AL:VA;0) 298.15
     +1.5*ALCRNI2+1.5*ALCR2NI-1.5*ALCR3-1.5*ALNI3;,,N 99DUP6 !
 PARA L(FCC_L12,AL,CR:NI:VA;1) 298.15
     -1.5*ALCR2NI+1.5*AL2CRNI-0.5*AL3NI+0.5*CR3NI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,NI:CR:VA;1) 298.15
     -1.5*ALCRNI2+1.5*AL2CRNI-0.5*AL3CR+0.5*CRNI3;,,N 99DUP6 !
 PARA L(FCC_L12,CR,NI:AL:VA;1) 298.15
     -1.5*ALCRNI2+1.5*ALCR2NI-0.5*ALCR3+0.5*ALNI3;,,N 99DUP6 !
$
$                                                                 SIGMA PHASE
$                                                                  metastable
 PARA G(SIGMA,NI:CR:AL;0)  298.15
    +8*GHSERNI+4*GHSERCR+18*GBCCAL+UNASS;,,   N 99LEE !
 PARA G(SIGMA,AL:CR:NI;0)  298.15
    +8*GHSERAL+4*GHSERCR+18*GBCCNI+UNASS;,,  N 99LEE !
$
$                                                              NEWSIGMA PHASE
$                                                                  metastable
 PARA G(NEWSIGMA,NI:CR:AL;0)  298.15
    +10*GHSERNI+4*GHSERCR+16*GBCCAL;,,   N LINEAR !
 PARA G(NEWSIGMA,AL:CR:NI;0)  298.15
    +10*GHSERAL+4*GHSERCR+16*GBCCNI;,,  N LINEAR !
$
$
$****************************************************************************
$ Diffusion parameters
$
PARAMETER MQ(FCC_A1&AL,NI;0) 298.15 -284000+R*T*LN(7.5E-4);
	6.000E+3 N 96Eng !
PARAMETER MQ(FCC_A1&AL,AL;0) 298.15 -142000+R*T*LN(1.71E-4); 
	6.000E+3 N 96Eng !
PARAMETER MQ(FCC_A1&AL,AL,NI;0) 298.15 -41300-91.2*T; 
	6.000E+3 N 96Eng !
PARAMETER MQ(FCC_A1&AL,CR;0) 298.15 -235000-82*T; 6000 N 96Eng !
PARAMETER MQ(FCC_A1&AL,CR,NI;0) 298.15 -53200; 6000 N 96Eng !
PARAMETER MQ(FCC_A1&AL,Al,Cr;0) 298.15 +335000; 6000 N 96Eng !

PARAMETER MQ(FCC_A1&NI,AL;0) 298.15 -145900+R*T*LN(4.4E-4);
	6.000E+3 N 96Eng !
PARAMETER MQ(FCC_A1&NI,NI;0)298.15 -287000-69.8*T; 
	6.000E+3 N 96Eng !
PARAMETER MQ(FCC_A1&NI,AL,NI;0) 298.15 -113000+65.5*T;
	6.000E+3 N 96Eng !
PARAMETER MQ(FCC_A1&NI,CR;0) 298.15 -235000-82*T; 6000 N 96Eng !
PARAMETER MQ(FCC_A1&NI,CR,NI;0) 298.15 -81000; 6000 N 96Eng !
PARAMETER MQ(FCC_A1&NI,AL,CR;0) 298.15 211000; 6000 N 96Eng !


PARAMETER MQ(FCC_A1&CR,AL;0) 298.15 -261700+R*T*LN(0.64); 
	6000 N 96Eng !
PARAMETER MQ(FCC_A1&CR,CR;0) 298.15 -235000-82*T; 6000 N 96Eng !
PARAMETER MQ(FCC_A1&CR,NI;0) 298.15 -287000-64.4*T; 6000 N 96Eng !
PARAMETER MQ(FCC_A1&CR,AL,CR;0) 298.15 +487000; 6000 N 96Eng !
PARAMETER MQ(FCC_A1&CR,AL,NI;0) 298.15 -118000; 6000 N 96Eng !
PARAMETER MQ(FCC_A1&CR,CR,NI;0) 298.15 -68000; 6000 N 96Eng !
$
$****************************************************************************

 LIST_OF_REFERENCES
 NUMBER  SOURCE
  LINEAR    'Unassessed parameter, linear combination of unary data. (MU,
           SIGMA)'
  REFLAV    'G(LAVES,AA)=3*G(SER,AA)+15000'
  86DIN     'A. Dinsdale, T. Chart, MTDS NPL, Unpublished work (1986); CR-NI'
  86FER1    'A. Fernandez Guillermet,
          Z metallkde, Vol 78 (1987) p 639-647,
          TRITA-MAC 324B (1986); CO-NI'
  86GUS     'P. Gustafson, Calphad Vol 11 (1987) p 277-292,
          TRITA-MAC 320 (1986); CR-NI-W '
  87GUS     'P. Gustafson, TRITA-MAC 342, (1987); CR-FE-W'
  89DIN     'Alan Dinsdale, SGTE Data for Pure Elements,
          NPL Report DMA(A)195 September 1989'
  91DIN     'Alan Dinsdale, SGTE Data for Pure Elements, NPL Report
          DMA(A)195 Rev. August 1990'
  91LEE     'Byeong-Joo Lee, unpublished revision (1991); C-Cr-Fe-Ni'
  91SAU1    'Nigel Saunders, 1991, based on
            N. Saunders, V.G. Rivlin
            Z. metallkde, 78 (11), 795-801 (1987); Al-Cr'
  91DIN     'Alan Dinsdale, SGTE Data for Pure Elements,
          Calphad Vol 15(1991) p 317-425, 
          also in NPL Report DMA(A)195 Rev. August 1990'
  95DUP3     'N. Dupin, Thesis, LTPCM, France, 1995; 
          Al-Ni,
          also in I. Ansara, N. Dupin, H.L. Lukas, B. SUndman
          J. Alloys Compds, 247 (1-2), 20-30 (1997)'
  95DUP4     'N. Dupin, Thesis, LTPCM, France, 1995;
          Cr-Ni-Ta'
  95DUP6     'N. Dupin, Thesis, LTPCM, France, 1995;
          Al-Cr-Ni'
  95DUP8     'N. Dupin, Thesis, LTPCM, France, 1995;
          Al-Ni-Ti'
  95DUP9     'N. Dupin, Thesis, LTPCM, France, 1995;
          Cr-Ni-Ti'
  99DUP      'N. Dupin, I. Ansara,
          Z. metallkd., Vol 90 (1999) p 76-85;
          Al-Ni'
  99DUP3    'N. Dupin, 
          July 1999, unpublished revision 
          ; Al-Ni'
  99DUP6    'N. Dupin, 
          July 1999, unpublished revision 
          ; Al-Cr-Ni'
  99DUP8    'N. Dupin,
          August 1999, unpublished revision
          ; Al-Ni-Ti'
  99DUP9    'N. Dupin,
          August 1999, unpublished revision
          ; Cr-Ni-Ti'
  99LEE     'Byeong-Joo Lee, unpublished work at KTH (1999);
          update of steel database'
   REF184   'AL1<G> CODATA KEY VALUES SGTE ** 
          ALUMINIUM <GAS> 
          Cp values similar in Codata Key Values and IVTAN Vol. 3'
   REF448   'AL2<G> CHATILLON(1992)
         Enthalpy of formation for Al1<g> taken from Codata Key Values.
         Enthalpy of form. from TPIS dissociation energy mean Value
         corrected with new fef from Sunil K.K. and Jordan K.D.
         (J.Phys. Chem. 92(1988)2774) ab initio calculations.'
   REF4465  'CR1<G> T.C.R.A.S. Class: 1 
         CHROMIUM <GAS>'
   REF4591  'CR2<G> T.C.R.A.S. Class: 6'
   REF7504  'NI1<G> T.C.R.A.S Class: 1
         Data provided by T.C.R.A.S. October 1996'
   REF7553  'NI2<G> T.C.R.A.S Class: 5 
         Data provided by T.C.R.A.S. October 1996'
  ! 
"""

NICRAL_TDB_DIFF = """

$ Same as NICRAL_TDB with the following changes
$ Shortened to include just FCC_A1, FCC_L12, LIQUID
$ Mobility parameters replaced with arbitrary diffusivity parameter to test diffusivity and tracer outputs
$


 ELEMENT /-   ELECTRON_GAS               .0000E+00   .0000E+00   .0000E+00!
 ELEMENT VA   VACUUM                     .0000E+00   .0000E+00   .0000E+00!
 ELEMENT AL   FCC_A1                    2.6982E+01  4.5773E+03  2.8322E+01!
 ELEMENT CR   BCC_A2                    5.1996E+01  4.0500E+03  2.3560E+01!
 ELEMENT NI   FCC_A1                    5.8690E+01  4.7870E+03  2.9796E+01!

 SPECIES AL2                         AL2!
 SPECIES CR2                         CR2!
 SPECIES NI2                         NI2!

 
 FUNCTION UNASS      298.15  0;,,N !


 TYPE_DEFINITION % SEQ *!
 DEFINE_SYSTEM_DEFAULT E 2 !
 DEFAULT_COMMAND DEF_SYS_ELEMENT VA !
 DEFAULT_COMMAND REJECT_PHASE NEWSIGMA !
			  
 TYPE_DEFINITION A GES A_P_D FCC_L12 MAGNETIC  -3.0 .28 !
 TYPE_DEFINITION E GES A_P_D FCC_A1 MAGNETIC  -3.0 .28 !

 TYPE_DEFINITION D GES A_P_D FCC_L12 DIS_PART FCC_A1 !

 PHASE GAS:G %  1  1.0  !
 CONST GAS:G :AL,AL2,CR,CR2,NI,NI2 :  !

 PHASE LIQUID:L %  1  1.0  !
 CONST LIQUID:L :AL,CR,NI :  !

 PHASE FCC_A1  %E  2 1   1 !
 CONST FCC_A1  :AL,CR,NI% : VA% :  !

$ PHASE FCC_L12  %ADG  3 .75   .25   1 !
 PHASE FCC_L12  %AD  3 .75   .25   1 !
 CONST FCC_L12  :AL,CR,NI : AL,CR,NI : VA :  !

 FUNCTION ZERO       298.15  0;,,N !
 FUNCTION DP         298.15  +P-101325;,,N !
 FUNCTION TROIS 298.15 3;,,N !
 FUNCTION UNTIER 298.15 TROIS**(-1);,,N !

$****************************************************************************
$
$                                                            UNARY PARAMETERS
$
$----------------------------------------------------------------------------
$
$                                                                          Al
$
$                                                                   FUNCTIONS
$
 FUNCTION F154T      298.15
    +323947.58-25.1480943*T-20.859*T*LN(T)
    +4.5665E-05*T**2-3.942E-09*T**3-24275.5*T**(-1);
                    4300.0  Y
    +342017.233-54.0526109*T-17.7891*T*LN(T)+6.822E-05*T**2
    -1.91111667E-08*T**3-14782200*T**(-1);
                    8200.0  Y
    +542396.07-411.214335*T+22.2419*T*LN(T)-.00349619*T**2
    +4.0491E-08*T**3-2.0366965E+08*T**(-1);  1.00000E+04  N !
$
 FUNCTION F625T      298.15
    +496408.232+35.479739*T-41.6397*T*LN(T)
    +.00249636*T**2-4.90507333E-07*T**3+85390.3*T**(-1);
                     900.00  Y
    +497613.221+17.368131*T-38.85476*T*LN(T)-2.249805E-04*T**2
    -9.49003167E-09*T**3-5287.23*T**(-1);  2.80000E+03  N !
$
 FUNCTION GHSERAL    298.15
    -7976.15+137.093038*T-24.3671976*T*LN(T)
    -.001884662*T**2-8.77664E-07*T**3+74092*T**(-1);
                     700.00  Y
    -11276.24+223.048446*T-38.5844296*T*LN(T)
    +.018531982*T**2-5.764227E-06*T**3+74092*T**(-1);
                     933.60  Y
    -11278.378+188.684153*T-31.748192*T*LN(T)
    -1.231E+28*T**(-9);,,  N !
$
 FUNCTION GHCPAL     298.15  +5481-1.8*T+GHSERAL;,,N !
$
 FUNCTION GBCCAL     298.15  +10083-4.813*T+GHSERAL;,,N !
$
 FUNCTION GLIQAL     298.14
    +11005.029-11.841867*T+7.934E-20*T**7+GHSERAL;
                     933.59  Y
    +10482.282-11.253974*T+1.231E+28*T**(-9)+GHSERAL;,,N !
$
$                                                                   GAS PHASE
$
 PARAMETER G(GAS,AL;0)  298.15  +F154T+R*T*LN(1E-05*P);,,N REF184 !
 PARAMETER G(GAS,AL2;0)  298.15  +F625T+R*T*LN(1E-05*P);,,N REF448 !
$
$                                                                LIQUID PHASE
$
 PARAMETER   G(LIQUID,AL;0)   298.13
      +11005.029-11.841867*T+7.934E-20*T**7+GHSERAL;
                                933.60  Y
      +10482.382-11.253974*T+1.231E+28*T**(-9)
      +GHSERAL;,,N 91DIN !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER G(FCC_A1,AL:VA;0)  298.15  +GHSERAL;,,N 91DIN !
$
$
$----------------------------------------------------------------------------
$
$                                                                          Cr
$
$                                                                   FUNCTIONS
$
 FUNCTION F7454T     298.15
    +390765.331-31.5192154*T-21.36083*T*LN(T)
    +7.253215E-04*T**2-1.588679E-07*T**3+10285.15*T**(-1);
                     1100.0  Y
    +393886.928-44.107465*T-19.96003*T*LN(T)+.001513089*T**2
    -4.23648333E-07*T**3-722515*T**(-1);
                     2000.0  Y
    +421372.003-231.888524*T+5.362886*T*LN(T)-.00848877*T**2
    +2.984635E-07*T**3-6015405*T**(-1);
                     3300.0  Y
    +305164.698+251.019831*T-55.20304*T*LN(T)+.005324585*T**2
    -2.850405E-07*T**3+34951485*T**(-1);
                     5100.0  Y
    +1069921.1-1708.93262*T+175.0508*T*LN(T)-.025574185*T**2
    +4.94447E-07*T**3-4.4276355E+08*T**(-1);
                     7600.0  Y
    -871952.838+1686.47356*T-204.5589*T*LN(T)+.007475225*T**2
    -4.618745E-08*T**3+1.423504E+09*T**(-1);  1.00000E+04  N !
$
 FUNCTION F7735T     298.15  +598511.402+41.5353219*T-40.56798*T*LN(T)
    +.004961847*T**2-1.61216717E-06*T**3+154422.85*T**(-1);
                     800.00  Y
    +613345.232-104.20799*T-19.7643*T*LN(T)-.007085085*T**2
    -4.69883E-07*T**3-1738066.5*T**(-1);
                     1400.0  Y
    +642608.843-369.286259*T+17.64743*T*LN(T)-.02767321*T**2
    +1.605906E-06*T**3-5831655*T**(-1);
                     2300.0  Y
    +553119.895+159.188556*T-52.07969*T*LN(T)-.004229401*T**2
    +1.5939925E-07*T**3+14793625*T**(-1);
                     3900.0  Y
    +347492.339+623.137624*T-105.0428*T*LN(T)+3.9699545E-04*T**2
    +1.51783483E-07*T**3+1.4843765E+08*T**(-1);
                     5800.0  Y
    -484185.055+2598.25559*T-334.7145*T*LN(T)+.028597625*T**2
    -4.97520167E-07*T**3+7.135805E+08*T**(-1);  6.00000E+03  N !
$
 FUNCTION GHSERCR    298.14
    -8856.94+157.48*T-26.908*T*LN(T)
    +.00189435*T**2-1.47721E-06*T**3+139250*T**(-1);
                     2180.0  Y
    -34869.344+344.18*T-50*T*LN(T)-2.88526E+32*T**(-9);,,N !
$
 FUNCTION GCRLIQ     298.15
    +24339.955-11.420225*T+2.37615E-21*T**7+GHSERCR;
                     2180.0  Y
    -16459.984+335.616316*T-50*T*LN(T);,,N !
$
 FUNCTION GFCCCR     298.15  +7284+.163*T+GHSERCR;,,N !
$
 FUNCTION GHCPCR     298.15  +4438+GHSERCR;,,N !
$
 FUNCTION ACRBCC     298.15  +1.7E-05*T+9.2E-09*T**2;,,N !
 FUNCTION BCRBCC     298.15  +1+2.6E-11*P;,,N !
 FUNCTION CCRBCC     298.15  2.08E-11;,,N !
 FUNCTION DCRBCC     298.15  +1*LN(BCRBCC);,,N !
 FUNCTION VCRBCC     298.15  +7.188E-06*EXP(ACRBCC);,,N !
 FUNCTION ECRBCC     298.15  +1*LN(CCRBCC);,,N !
 FUNCTION XCRBCC     298.15  +1*EXP(.8*DCRBCC)-1;,,N !
 FUNCTION YCRBCC     298.15  +VCRBCC*EXP(-ECRBCC);,,N !
 FUNCTION ZCRBCC     298.15  +1*LN(XCRBCC);,,N !
 FUNCTION GPCRBCC    298.15  +YCRBCC*EXP(ZCRBCC);,,N !
$
 FUNCTION ACRLIQ     298.15  +1.7E-05*T+9.2E-09*T**2;,,N !
 FUNCTION BCRLIQ     298.15  +1+4.65E-11*P;,,N !
 FUNCTION CCRLIQ     298.15  3.72E-11;,,N !
 FUNCTION DCRLIQ     298.15  +1*LN(BCRLIQ);,,N !
 FUNCTION VCRLIQ     298.15  +7.653E-06*EXP(ACRLIQ);,,N !
 FUNCTION ECRLIQ     298.15  +1*LN(CCRLIQ);,,N !
 FUNCTION XCRLIQ     298.15  +1*EXP(.8*DCRLIQ)-1;,,N !
 FUNCTION YCRLIQ     298.15  +VCRLIQ*EXP(-ECRLIQ);,,N !
 FUNCTION ZCRLIQ     298.15  +1*LN(XCRLIQ);,,N !
 FUNCTION GPCRLIQ    298.15  +YCRLIQ*EXP(ZCRLIQ);,,N !
$
$                                                                   GAS PHASE
$
 PARAMETER G(GAS,CR;0)  298.15  +F7454T+R*T*LN(1E-05*P);,,N REF4465 !
 PARAMETER G(GAS,CR2;0)  298.15  +F7735T+R*T*LN(1E-05*P);,,  N REF4591 !
$
$                                                                LIQUID PHASE
$
 PARAMETER G(LIQUID,CR;0)  298.15  +GCRLIQ+GPCRLIQ;,,  N 91DIN !
$
$                                                                FCC_A1 PHASE
$
$
$----------------------------------------------------------------------------
$
$                                                                          Ni
$
$                                                                   FUNCTIONS
$
 FUNCTION F13191T    298.15
    +417658.868-44.7777921*T-20.056*T*LN(T)
    -.0060415*T**2+1.24774E-06*T**3-16320*T**(-1);
                     800.00  Y
    +413885.448+9.41787679*T-28.332*T*LN(T)+.00173115*T**2
    -8.399E-08*T**3+289050*T**(-1);
                     3900.0  Y
    +440866.732-62.5810038*T-19.819*T*LN(T)+5.067E-04*T**2
    -4.93233333E-08*T**3-15879735*T**(-1);
                     7600.0  Y
    +848806.287-813.398164*T+64.69*T*LN(T)-.00731865*T**2
    +8.71833333E-08*T**3-3.875846E+08*T**(-1);  10000.  N !
$
 FUNCTION F13265T    298.15
    +638073.279-68.1901928*T-24.897*T*LN(T)
    -.0313584*T**2+5.93355333E-06*T**3-14215*T**(-1);
                     800.00  Y
    +611401.772+268.084821*T-75.25401*T*LN(T)+.01088525*T**2
    -7.08741667E-07*T**3+2633835*T**(-1);
                     2100.0  Y
    +637459.339+72.0712678*T-48.587*T*LN(T)-9.09E-05*T**2
    +9.12933333E-08*T**3-1191755*T**(-1);
                     4500.0 Y
    +564540.781+329.599011*T-80.11301*T*LN(T)+.00578085*T**2
    -1.08841667E-07*T**3+29137900*T**(-1);  6000.0  N !
$
 FUNCTION GHSERNI    298.14
    -5179.159+117.854*T-22.096*T*LN(T)
    -.0048407*T**2;
                     1728.0  Y
    -27840.655+279.135*T-43.1*T*LN(T)+1.12754E+31*T**(-9);,,  N   !
$
 FUNCTION GHCPNI     298.15  +1046+1.2552*T+GHSERNI;,,N !
$
 FUNCTION GBCCNI     298.15  +8715.084-3.556*T+GHSERNI;,,,   N !
$
$                                                                   GAS PHASE
$
 PARAMETER G(GAS,NI;0)  298.15  +F13191T+R*T*LN(1E-05*P);,,N REF7504 !
 PARAMETER G(GAS,NI2;0)  298.15 +F13265T+R*T*LN(1E-05*P);,,N REF7553 !
$
$                                                                LIQUID PHASE
$
 PARAMETER G(LIQUID,NI;0) 298.13
      +16414.686-9.397*T-3.82318E-21*T**7+GHSERNI;
                            1728.0  Y
      +18290.88-10.537*T-1.12754E+31*T**(-9)
      +GHSERNI;,,N 91DIN !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER G(FCC_A1,NI:VA;0)  298.15  +GHSERNI;,,N 91DIN !
 PARAMETER TC(FCC_A1,NI:VA;0)  298.15  633;,,N 89DIN !
 PARAMETER BMAGN(FCC_A1,NI:VA;0)  298.15  .52;,,N 89DIN !
$
$
$****************************************************************************
$
$                                                           BINARY PARAMETERS
$
$----------------------------------------------------------------------------
$
$                                                                       Al-Cr
$                             Mainly from Saunders (COST507)
$                             Metastable B2 and L12 from revision of Al-Cr-Ni
$
$                                                                LIQUID PHASE
$
 PARAMETER L(LIQUID,AL,CR;0)  298.15  -29000;,,N 91SAU1 !
 PARAMETER L(LIQUID,AL,CR;1)  298.15  -11000;,,N 91SAU1 !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER G(FCC_A1,AL,CR:VA;0)  298.15  -45900+6*T;,,N 91SAU1 !
$
$
$                                                               FCC_L12 PHASE
$                                                                  metastable
$ Present work: july 1999, study of Al-Cr-Ni, revision of NDTH.
     FUN U1ALCR  298.15 -830;,,N  99DUP6 !
     FUN U3ALCR  298.15 0.0; 6000.00   99DUP6 !
     FUN U4ALCR  298.15 0.0; 6000.00 N  99DUP6 !
   FUNCTION L04ALCR 298.15 U3ALCR;,,N !
   FUNCTION L14ALCR 298.15 U4ALCR;,,N !
   FUNCTION ALCR3 298.15 3*U1ALCR;,,N !
   FUNCTION AL2CR2 298.15 4*U1ALCR;,,N !
   FUNCTION AL3CR 298.15 3*U1ALCR;,,N !
 PARAMETER G(FCC_L12,CR:AL:VA;0)  298.15  +ALCR3;,,  N  99DUP6 !
 PARAMETER G(FCC_L12,AL:CR:VA;0)  298.15  +AL3CR;,,  N  99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:AL:VA;0) 298.15
     -1.5*ALCR3+1.5*AL2CR2+1.5*AL3CR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:CR:VA;0) 298.15
     +1.5*ALCR3+1.5*AL2CR2-1.5*AL3CR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:AL:VA;1) 298.15
     +0.5*ALCR3-1.5*AL2CR2+1.5*AL3CR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:CR:VA;1) 298.15
     -1.5*ALCR3+1.5*AL2CR2-0.5*AL3CR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,*:AL,CR:VA;0) 298.15 +L04ALCR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,*:AL,CR:VA;1) 298.15 +L14ALCR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:*:VA;0) 298.15 +3*L04ALCR;,,N 99DUP6 !
 PARAMETER L(FCC_L12,AL,CR:*:VA;1) 298.15 +3*L14ALCR;,,N 99DUP6 !
$
$
$----------------------------------------------------------------------------
$
$                                                                       Al-Ni
$                     Mainly from ND thesis,
$                     slighly revised to get better solvus at low temperature
$
$                                                                LIQUID PHASE
$
 PARAMETER L(LIQUID,AL,NI;0)  298.15 -207109.28+41.31501*T;,,N 95DUP3 !
 PARAMETER L(LIQUID,AL,NI;1)  298.15 -10185.79+5.8714*T;,,N 95DUP3 !
 PARAMETER L(LIQUID,AL,NI;2)  298.15 +81204.81-31.95713*T;,,N 95DUP3 !
 PARAMETER L(LIQUID,AL,NI;3)  298.15  +4365.35-2.51632*T;,,N 95DUP3 !
 PARAMETER L(LIQUID,AL,NI;4)  298.15  -22101.64+13.16341*T;,,N 95DUP3 !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER TC(FCC_A1,AL,NI:VA;0)  298.15  -1112;,,N 95DUP3 !
 PARAMETER TC(FCC_A1,AL,NI:VA;1)  298.15  1745;,,N 95DUP3 !
 PARAMETER G(FCC_A1,AL,NI:VA;0)  298.15  -162407.75+16.212965*T;,,N 95DUP3 !
 PARAMETER G(FCC_A1,AL,NI:VA;1)  298.15  +73417.798-34.914168*T;,,N 95DUP3 !   
 PARAMETER G(FCC_A1,AL,NI:VA;2)  298.15  +33471.014-9.8373558*T;,,N 95DUP3 !   
 PARAMETER G(FCC_A1,AL,NI:VA;3)  298.15  -30758.01+10.25267*T;,,N 95DUP3 !
$
$                                                               FCC_L12 PHASE
$
     FUN UALNI 298.15 -22212.8931+4.39570389*T;,,,N 99DUP3 !
     FUN U1ALNI 298.15 2*UNTIER*UALNI;,,,N 99DUP3 !
     FUN U3ALNI 298.15 0;,,,N 99DUP3 !
     FUN U4ALNI 298.15 7203.60609-3.74273030*T;,,,N 99DUP3 !
   FUNCTION L04ALNI 298.15 U3ALNI;,,N 99DUP3 !
   FUNCTION L14ALNI 298.15 U4ALNI;,,N 99DUP3 !
   FUNCTION ALNI3   298.15 +3*U1ALNI;,,,N 99DUP3 !
   FUNCTION AL2NI2  298.15 +4*U1ALNI;,,,N 99DUP3 !
   FUNCTION AL3NI   298.15 +3*U1ALNI;,,,N 99DUP3 !
 PARAMETER G(FCC_L12,NI:AL:VA;0)  298.15  +ALNI3;,,N 99DUP3 !
 PARAMETER G(FCC_L12,AL:NI:VA;0)  298.15  +AL3NI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:AL:VA;0) 298.15
     -1.5*ALNI3+1.5*AL2NI2+1.5*AL3NI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:NI:VA;0) 298.15
     +1.5*ALNI3+1.5*AL2NI2-1.5*AL3NI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:AL:VA;1) 298.15
     +0.5*ALNI3-1.5*AL2NI2+1.5*AL3NI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:NI:VA;1) 298.15
     -1.5*ALNI3+1.5*AL2NI2-0.5*AL3NI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,*:AL,NI:VA;0) 298.15 +L04ALNI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,*:AL,NI:VA;1) 298.15 +L14ALNI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:*:VA;0) 298.15 +3*L04ALNI;,,N 99DUP3 !
 PARAMETER L(FCC_L12,AL,NI:*:VA;1) 298.15 +3*L14ALNI;,,N 99DUP3 !
$
$
$----------------------------------------------------------------------------
$
$                                                                       Cr-Ni
$                             Mainly from SSOL
$                             Metastable B2 and L12 from revision of Al-Cr-Ni
$ 
$                                                                LIQUID PHASE
$
 PARAMETER L(LIQUID,CR,NI;0)  298.15  +318-7.3318*T;,,N 91LEE !
 PARAMETER L(LIQUID,CR,NI;1)  298.15  +16941-6.3696*T;,,N 91LEE !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER G(FCC_A1,CR,NI:VA;0)  298.15  +8030-12.8801*T;,,N 91LEE !
 PARAMETER G(FCC_A1,CR,NI:VA;1)  298.15  +33080-16.0362*T;,,N 91LEE !   
 PARAMETER TC(FCC_A1,CR,NI:VA;0)  298.15  -3605;,,N 86DIN !
 PARAMETER BMAGN(FCC_A1,CR,NI:VA;0)  298.15  -1.91;,,N 86DIN !
$
$
$                                                               FCC_L12 PHASE
$                                                                  metastable
$ Present work: july 1999, study of Al-Cr-Ni, revision of NDTH.
$ The L12 phase is metastable in the binary Cr-Ni while it was stable in NDTH.
     FUN U1CRNI 298.15 -1980;,,,N 99DUP6 !
$     FUN U1CRNI 298.15 -7060+3.63*T;,,,N 99DUP6 !
     FUN U3CRNI 298.15 0;,,,N 99DUP6 !
     FUN U4CRNI 298.15 0;,,,N 99DUP6 !
   FUNCTION L04CRNI 298.15 U3CRNI;,,N 99DUP6 !
   FUNCTION L14CRNI 298.15 U4CRNI;,,N 99DUP6 !
   FUNCTION CRNI3   298.15 +3*U1CRNI;,,,N 99DUP6 !
   FUNCTION CR2NI2  298.15 +4*U1CRNI;,,,N 99DUP6 !
   FUNCTION CR3NI   298.15 +3*U1CRNI;,,,N 99DUP6 !
 PARAMETER G(FCC_L12,NI:CR:VA;0)  298.15  +CRNI3;,,  N 99DUP6 !
 PARAMETER G(FCC_L12,CR:NI:VA;0)  298.15  +CR3NI;,,  N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:CR:VA;0) 298.15
     -1.5*CRNI3+1.5*CR2NI2+1.5*CR3NI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:NI:VA;0) 298.15
     +1.5*CRNI3+1.5*CR2NI2-1.5*CR3NI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:CR:VA;1) 298.15
     +0.5*CRNI3-1.5*CR2NI2+1.5*CR3NI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:NI:VA;1) 298.15
     -1.5*CRNI3+1.5*CR2NI2-0.5*CR3NI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,*:CR,NI:VA;0) 298.15 +L04CRNI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,*:CR,NI:VA;1) 298.15 +L14CRNI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:*:VA;0) 298.15 +3*L04CRNI;,,N 99DUP6 !
 PARAMETER L(FCC_L12,CR,NI:*:VA;1) 298.15 +3*L14CRNI;,,N 99DUP6 !
$                                                        
$
$****************************************************************************
$
$                                                          TERNARY PARAMETERS
$
$----------------------------------------------------------------------------
$
$                                                                    Al-Cr-Ni
$                                    July 1999, ND
$                                    Revision. Main changes:
$                                    - description of the A2/B2
$                                    - new liquidus data taken into account
$                                    - simpler ternary interaction parameters
$
$                                                                LIQUID PHASE
$
 PARAMETER L(LIQUID,AL,CR,NI;0)  298.15  16000;,,N 99DUP6 !
$
$                                                                FCC_A1 PHASE
$
 PARAMETER G(FCC_A1,AL,CR,NI:VA;0)  298.15  30300;,,N 99DUP6 !
$
$
$                                                               FCC_L12 PHASE
$
   FUN U1ALCRNI 298.15 6650;,,N 99DUP6 !
   FUN U2ALCRNI 298.15 0;,,N 99DUP6 !
   FUN U3ALCRNI 298.15 0;,,N 99DUP6 !
   FUN ALCRNI2 298.15 U1ALCR+2*U1ALNI+2*U1CRNI+U1ALCRNI;,,N 99DUP6 !
   FUN ALCR2NI 298.15 2*U1ALCR+U1ALNI+2*U1CRNI+U2ALCRNI;,,N 99DUP6 !
   FUN AL2CRNI 298.15 2*U1ALCR+2*U1ALNI+U1CRNI+U3ALCRNI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,CR,NI:AL:VA;0) 298.15
     -1.5*ALCRNI2-1.5*ALCR2NI+ALCR3+ALNI3+6*AL2CRNI
     -1.5*AL2CR2-1.5*AL2NI2-1.5*AL3CR-1.5*AL3NI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,CR,NI:CR:VA;0) 298.15
     -1.5*ALCRNI2+6*ALCR2NI-1.5*ALCR3-1.5*AL2CRNI
     -1.5*AL2CR2+AL3CR+CRNI3-1.5*CR2NI2-1.5*CR3NI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,CR,NI:NI:VA;0) 298.15
     +6*ALCRNI2-1.5*ALCR2NI-1.5*ALNI3-1.5*AL2CRNI
     -1.5*AL2NI2+AL3NI-1.5*CRNI3-1.5*CR2NI2+CR3NI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,CR:NI:VA;0) 298.15
     +1.5*ALCR2NI+1.5*AL2CRNI-1.5*AL3NI-1.5*CR3NI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,NI:CR:VA;0) 298.15
     +1.5*ALCRNI2+1.5*AL2CRNI-1.5*AL3CR-1.5*CRNI3;,,N 99DUP6 !
 PARA L(FCC_L12,CR,NI:AL:VA;0) 298.15
     +1.5*ALCRNI2+1.5*ALCR2NI-1.5*ALCR3-1.5*ALNI3;,,N 99DUP6 !
 PARA L(FCC_L12,AL,CR:NI:VA;1) 298.15
     -1.5*ALCR2NI+1.5*AL2CRNI-0.5*AL3NI+0.5*CR3NI;,,N 99DUP6 !
 PARA L(FCC_L12,AL,NI:CR:VA;1) 298.15
     -1.5*ALCRNI2+1.5*AL2CRNI-0.5*AL3CR+0.5*CRNI3;,,N 99DUP6 !
 PARA L(FCC_L12,CR,NI:AL:VA;1) 298.15
     -1.5*ALCRNI2+1.5*ALCR2NI-0.5*ALCR3+0.5*ALNI3;,,N 99DUP6 !
$
$****************************************************************************
$ Diffusion parameters
$
PARAMETER DQ(FCC_A1&AL,*;0) 298.15  -70000+R*T*LN(5E-5); 6000 N !
PARAMETER DQ(FCC_A1&CR,*;0) 298.15  -40800+R*T*LN(3e-6); 6000 N !
PARAMETER DQ(FCC_A1&NI,*;0) 298.15  -271960+R*T*LN(1.27E-4); 6000 N !
$
"""