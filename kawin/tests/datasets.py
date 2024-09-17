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

ALZR_TDB_NO_MOB = """
$Al-Zr database without any mobility parameters
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

FECRNI_DB = """
$ FeCrNi database using parameters taken from MatCalc open steel database mc_fe_v2.060.tdb
$
$ The mc_fe_v2.059.tdb database is made available under the 
$ Open Database License: http://opendatacommons.org/licenses/odbl/1.0/. 
$ Any rights in individual contents of the database are licensed under the 
$ Database Contents License: http://opendatacommons.org/licenses/dbcl/1.0/. 
$
$ ##########################################################################

ELEMENT VA   VACUUM            0.0                0.00            0.00      !
ELEMENT CR   BCC_A2           51.996           4050.0            23.5429    !
ELEMENT FE   BCC_A2           55.847           4489.0            27.2797    !
ELEMENT NI   FCC_A1           58.69            4787.0            29.7955    !

FUNCTION GHSERCR
 273.00 -8856.94+157.48*T-26.908*T*LN(T)
 +0.00189435*T**2-1.47721E-6*T**3+139250*T**(-1); 2180.00  Y
 -34869.344+344.18*T-50*T*LN(T)-2.88526E+32*T**(-9); 6000.00  N
REF:0 !
FUNCTION GCRFCC
 273.00 +7284+0.163*T+GHSERCR#; 6000.00  N
REF:0 !
FUNCTION GHSERFE
 273.00 +1225.7+124.134*T-23.5143*T*LN(T)-0.00439752*T**2
 -5.89269E-8*T**3+77358.5*T**(-1); 1811.00  Y
 -25383.581+299.31255*T-46*T*LN(T)+2.2960305E+31*T**(-9); 6000.00  N
REF:0 !
FUNCTION GFEFCC
 273.00 -1462.4+8.282*T-1.15*T*LN(T)+6.4E-04*T**2+GHSERFE#; 1811.00  Y
 -27098.266+300.25256*T-46*T*LN(T)+2.78854E+31*T**(-9); 6000.00  N
REF:0 !
FUNCTION GHSERNI
 273.00 -5179.159+117.854*T-22.096*T*LN(T)-4.8407E-3*T**2; 1728.00  Y
 -27840.655+279.135*T-43.10*T*LN(T)+1.12754E+31*T**(-9); 6000.00  N
REF:0 !
FUNCTION GNIBCC
 273.00 +8715.084-3.556*T+GHSERNI#; 6000.00  N
REF:0 !

$FCC_A1 phase

TYPE_DEFINITION ' GES A_P_D FCC_A1 MAGNETIC  -3.0    0.28 !
TYPE_DEFINITION % SEQ *!
PHASE FCC_A1  %'  2 1   1 ! 
CONSTITUENT FCC_A1  : CR,FE%,NI : VA% :  !

PARAMETER G(FCC_A1,CR:VA;0) 273.00 +7284+0.163*T+GHSERCR#; 6000.00  N
REF:0 !
PARAMETER G(FCC_A1,FE:VA;0) 273.00 -1462.4+8.282*T-1.15*T*LN(T)
   +0.00064*T**2+GHSERFE#; 1811.00  Y
   -1713.815+0.94001*T+0.4925095E+31*T**(-9)+GHSERFE#; 6000.00  N
REF:0 !
PARAMETER G(FCC_A1,NI:VA;0) 273.00 +GHSERNI#; 3000.00  N
REF:0 !
PARAMETER L(FCC_A1,CR,FE:VA;0) 273.00 +10833-7.477*T; 6000.00  N
REF:11 !
PARAMETER L(FCC_A1,CR,FE:VA;1) 273.00 +1410; 6000.00  N
REF:11 !
PARAMETER L(FCC_A1,CR,NI:VA;0) 273.00 +8030-12.8801*T; 6000.00  N
REF:11 !
PARAMETER L(FCC_A1,CR,NI:VA;1) 273.00 +33080-16.0362*T; 6000.00  N
REF:11 !
PARAMETER L(FCC_A1,FE,NI:VA;0) 273.00 -12054.355+3.27413*T; 6000.00  N
REF:20 !
PARAMETER L(FCC_A1,FE,NI:VA;1) 273.00 +11082.1315-4.45077*T; 6000.00  N
REF:20 !
PARAMETER L(FCC_A1,FE,NI:VA;2) 273.00 -725.805174; 6000.00  N
REF:20 !
PARAMETER L(FCC_A1,CR,FE,NI:VA;0) 273.00 +8000-8*T; 6000.00 N
REF:jac17 !
PARAMETER L(FCC_A1,CR,FE,NI:VA;1) 273.00 -6500; 6000.00 N
REF:jac17 !
PARAMETER L(FCC_A1,CR,FE,NI:VA;2) 273.00 +30000; 6000.00 N
REF:jac17 !
PARAMETER TC(FCC_A1,CR:VA;0) 273.00 -1109; 6000.00  N
REF:11 !
PARAMETER BMAGN(FCC_A1,CR:VA;0) 273.00 -2.46; 6000.00  N
REF:11 !
PARAMETER TC(FCC_A1,CR,NI:VA;0) 273.00 -3605; 6000.00  N
REF:11 !
PARAMETER BMAGN(FCC_A1,CR,NI:VA;0) 273.00 -1.91; 6000.00  N
REF:11 !
PARAMETER TC(FCC_A1,FE:VA;0) 273.00 -201; 6000.00  N
REF:20 !
PARAMETER BMAGN(FCC_A1,FE:VA;0) 273.00 -2.1; 6000.00  N
REF:20 !
PARAMETER TC(FCC_A1,FE,NI:VA;0) 273.00 +2133; 6000.00  N
REF:20 !
PARAMETER TC(FCC_A1,FE,NI:VA;1) 273.00 -682; 6000.00  N
REF:20 !
PARAMETER BMAGN(FCC_A1,FE,NI:VA;0) 273.00 +9.55; 6000.00  N
REF:20 !
PARAMETER BMAGN(FCC_A1,FE,NI:VA;1) 273.00 +7.23; 6000.00  N
REF:20 !
PARAMETER BMAGN(FCC_A1,FE,NI:VA;2) 273.00 +5.93; 6000.00  N
REF:20 !
PARAMETER BMAGN(FCC_A1,FE,NI:VA;3) 273.00 +6.18; 6000.00  N
REF:20 !
PARAMETER TC(FCC_A1,NI:VA;0) 273.00 +633; 6000.00  N
REF:0 !
PARAMETER BMAGN(FCC_A1,NI:VA;0) 273.00 +0.52; 6000.00  N
REF:0 !

$BCC_A2 phase

TYPE_DEFINITION & GES A_P_D BCC_A2 MAGNETIC  -1.0    0.4 !
PHASE BCC_A2  %&  2 1   3 !
CONSTITUENT BCC_A2  : CR,FE%,NI : VA% :  ! 

PARAMETER G(BCC_A2,CR:VA;0) 273.00 +GHSERCR#; 6000.00  N
REF:0 !
PARAMETER G(BCC_A2,FE:VA;0) 273.00 +GHSERFE#; 6000.00  N
REF:0 !
PARAMETER G(BCC_A2,NI:VA;0) 273.00 +8715.084-3.556*T+GHSERNI#; 3000.00  N
REF:0 !
PARAMETER L(BCC_A2,CR,FE:VA;0) 273.00 +20500-9.68*T; 6000.00  N
REF:11 !
PARAMETER L(BCC_A2,CR,NI:VA;0) 273.00 +17170-11.8199*T; 6000.00  N
REF:11 !
PARAMETER L(BCC_A2,CR,NI:VA;1) 273.00 +34418-11.8577*T; 6000.00  N
REF:11 !
PARAMETER L(BCC_A2,CR,NI:VA;2) 273.00 +1e-8; 6000.00  N
REF:11 !
PARAMETER L(BCC_A2,FE,NI:VA;0) 273.00 -956.63-1.28726*T; 6000.00  N
REF:20 !
PARAMETER L(BCC_A2,FE,NI:VA;1) 273.00 +5000-5*T; 6000.00  N
REF:pov12 !
PARAMETER L(BCC_A2,CR,FE,NI:VA;0) 273.00 +3000+5*T; 6000.00 N
REF:jac17 !
PARAMETER L(BCC_A2,CR,FE,NI:VA;1) 273.00 +9000-6*T; 6000.00 N
REF:jac17 !
PARAMETER L(BCC_A2,CR,FE,NI:VA;2) 273.00 -30000+20*T; 6000.00 N
REF:jac17 !
PARAMETER TC(BCC_A2,CR:VA;0) 273.00 -311.5; 6000.00  N
REF:22 !
PARAMETER BMAGN(BCC_A2,CR:VA;0) 273.00 -0.008; 6000.00  N
REF:0 !
PARAMETER TC(BCC_A2,FE:VA;0) 273.00 +1043; 6000.00  N
REF:0 !
PARAMETER BMAGN(BCC_A2,FE:VA;0) 273.00 +2.22; 6000.00  N
REF:0 !
PARAMETER TC(BCC_A2,NI:VA;0) 273.00 +575; 6000.00  N
REF:0 !
PARAMETER BMAGN(BCC_A2,NI:VA;0) 273.00 +0.85; 6000.00  N
REF:0 !
PARAMETER TC(BCC_A2,CR,FE:VA;0) 273.00 +1650; 6000.00  N
REF:11 !
PARAMETER TC(BCC_A2,CR,FE:VA;1) 273.00 +550; 6000.00  N
REF:11 !
PARAMETER BMAGN(BCC_A2,CR,FE:VA;0) 273.00 -0.85; 6000.00  N
REF:pov09 !
PARAMETER TC(BCC_A2,CR,NI:VA;0) 273.00 +2373; 6000.00  N
REF:11 !
PARAMETER TC(BCC_A2,CR,NI:VA;1) 273.00 +617; 6000.00  N
REF:11 !
PARAMETER BMAGN(BCC_A2,CR,NI:VA;0) 273.00 +4; 6000.00  N
REF:11 !

$SIGMA phase

PHASE SIGMA %  3 8   4   18  !
CONSTITUENT SIGMA  : FE%,NI : CR% : CR,FE,NI :!

PARAMETER G(SIGMA,FE:CR:CR;0) 273.00 +8*GFEFCC#+22*GHSERCR#
 +92300-95.96*T; 6000.00  N
REF:62 !
PARAMETER G(SIGMA,FE:CR:FE;0) 273.00 +117300-95.96*T+8*GFEFCC#
   +4*GHSERCR#+18*GHSERFE#; 6000.00  N
REF:62 !
PARAMETER G(SIGMA,FE:CR:NI;0) 273.00 -50000+32*T+8*GFEFCC#+4*GHSERCR#
   +18*GNIBCC#; 6000.00  N
REF:pov13 !
PARAMETER G(SIGMA,NI:CR:CR;0) 273.00 +8*GHSERNI#+22*GHSERCR#
 +180000-170*T; 6000.00  N
REF:13 !
PARAMETER G(SIGMA,NI:CR:FE;0) 273.00 +8*GHSERNI#+4*GHSERCR#
 +18*GHSERFE#-50000+32*T; 6000.00  N
REF:pov13 !
PARAMETER G(SIGMA,NI:CR:NI;0) 273.00 +8*GHSERNI#+4*GHSERCR#
 +18*GNIBCC#+175400; 6000.00  N
REF:13 !
PARAMETER L(SIGMA,FE:CR:CR,NI;0) 273.00 +1e-8; 6000.00  N
REF:pov12 !
PARAMETER L(SIGMA,FE:CR:FE,NI;0) 273.00 -200000; 6000.00  N
REF:pov13 !

$FCC mobility
$CR

PARAMETER MQ(FCC_A1&CR,CR:*) 273.00 -235000-82.0*T; 6000.00  N
Ref:19 !
PARAMETER MQ(FCC_A1&CR,FE:*) 273.00 -286000-71.9*T; 6000.00  N
Ref:19 !
PARAMETER MQ(FCC_A1&CR,NI:*) 273.00 -287000-64.4*T; 6000.00  N
Ref:19 !
PARAMETER MQ(FCC_A1&CR,CR,FE:*;0) 273.00 -105000; 6000.00  N
Ref:19 !
PARAMETER MQ(FCC_A1&CR,CR,NI:*;0) 273.00 -68000; 6000.00  N
Ref:41 !
PARAMETER MQ(FCC_A1&CR,FE,NI:*;0) 273.00 +16100; 6000.00  N
Ref:17 !
PARAMETER MQ(FCC_A1&CR,CR,FE,NI:*;0) 273.00 +310000; 6000.00  N
Ref:17 !
PARAMETER MQ(FCC_A1&CR,CR,FE,NI:*;1) 273.00 +320000; 6000.00  N
Ref:17 !
PARAMETER MQ(FCC_A1&CR,CR,FE,NI:*;2) 273.00 +120000; 6000.00  N
Ref:17 !

$FE

PARAMETER MQ(FCC_A1&FE,CR:*) 273.00 -235000-82.0*T; 6000.00  N
Ref:19 !
PARAMETER MQ(FCC_A1&FE,FE:*) 273.00 -286000+R*T*LN(7.0E-5); 6000.00  N
Ref:18 !
PARAMETER MQ(FCC_A1&FE,NI:*) 273.00 -287000-67.5*T; 6000.00  N
Ref:18 !
PARAMETER MQ(FCC_A1&FE,CR,FE:*;0) 273.00 +15900; 6000.00  N
Ref:19 !
PARAMETER MQ(FCC_A1&FE,CR,NI:*;0) 273.00 -77500; 6000.00  N
Ref:17 !
PARAMETER MQ(FCC_A1&FE,FE,NI:*;0) 273.00 -115000+104*T; 6000.00  N
Ref:18 !
PARAMETER MQ(FCC_A1&FE,FE,NI:*;1) 273.00 +78800-73.3*T; 6000.00  N
Ref:18 !
PARAMETER MQ(FCC_A1&FE,CR,FE,NI:*;0) 273.00 -740000; 6000.00  N
Ref:17 !
PARAMETER MQ(FCC_A1&FE,CR,FE,NI:*;1) 273.00 -540000; 6000.00  N
Ref:17 !
PARAMETER MQ(FCC_A1&FE,CR,FE,NI:*;2) 273.00 +750000; 6000.00  N
Ref:17 !

$NI

PARAMETER MQ(FCC_A1&NI,CR:*) 273.00 -235000-82.0*T; 6000.00  N
Ref:19 !
PARAMETER MQ(FCC_A1&NI,FE:*) 273.00 -286000-86.0*T; 6000.00  N
Ref:18 !
PARAMETER MQ(FCC_A1&NI,NI:*) 273.00 -287000-69.8*T; 6000.00  N
Ref:18 !
PARAMETER MQ(FCC_A1&NI,CR,FE:*;0) 273.00 -119000; 6000.00  N
Ref:17 !
PARAMETER MQ(FCC_A1&NI,CR,NI:*;0) 273.00 -81000; 6000.00  N
Ref:19 !
PARAMETER MQ(FCC_A1&NI,FE,NI:*;0) 273.00 +124000-51.4*T; 6000.00  N
Ref:18 !
PARAMETER MQ(FCC_A1&NI,FE,NI:*;1) 273.00 -300000+213*T; 6000.00  N
Ref:18 !
PARAMETER MQ(FCC_A1&NI,CR,FE,NI:*;0) 273.00 +1840000; 6000.00  N
Ref:17 !
PARAMETER MQ(FCC_A1&NI,CR,FE,NI:*;1) 273.00 +670000; 6000.00  N
Ref:17 !
PARAMETER MQ(FCC_A1&NI,CR,FE,NI:*;2) 273.00 -1120000; 6000.00  N
Ref:17 !

$BCC Mobility

$CR

PARAMETER MQ(BCC_A2&CR,CR:*) 273.00 -407000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,CR:*) 273.00 -35.6*T; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&CR,FE:*) 273.00 -218000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,FE:*) 273.00 +R*T*LN(8.5E-5); 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&CR,NI:*) 273.00 -218000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,NI:*) 273.00 +R*T*LN(8.5E-5); 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&CR,CR,FE:*;0) 273.00 +361000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,CR,FE:*;0) 273.00 -116*T; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&CR,CR,FE:*;1) 273.00 +2820; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,CR,FE:*;1) 273.00 +37.5*T; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&CR,CR,NI:*;0) 273.00 +350000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,CR,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&CR,FE,NI:*;0) 273.00 +150000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,FE,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&CR,FE,NI:*;1) 273.00 +150000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,FE,NI:*;1) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&CR,FE,NI:*;2) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,FE,NI:*;2) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&CR,CR,FE,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,CR,FE,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&CR,CR,FE,NI:*;1) 273.00 -2400000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,CR,FE,NI:*;1) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&CR,CR,FE,NI:*;2) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&CR,CR,FE,NI:*;2) 273.00 +1e-8; 6000.00  N
Ref:14 !

$FE

PARAMETER MQ(BCC_A2&FE,CR:*) 273.00 -407000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&FE,CR:*) 273.00 -17.2*T; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&FE,FE:*) 273.00 -218000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&FE,FE:*) 273.00 +R*T*LN(4.6E-5); 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&FE,NI:*) 273.00 -218000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&FE,NI:*) 273.00 +R*T*LN(4.6E-5); 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&FE,CR,FE:*;0) 273.00 +267000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&FE,CR,FE:*;0) 273.00 -117*T; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&FE,CR,FE:*;1) 273.00 -416000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&FE,CR,FE:*;1) 273.00 +348*T; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&FE,CR,NI:*;0) 273.00 +350000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&FE,CR,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&FE,FE,NI:*;0) 273.00 +150000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&FE,FE,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&FE,CR,FE,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&FE,CR,FE,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&FE,CR,FE,NI:*;1) 273.00 +1400000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&FE,CR,FE,NI:*;1) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&FE,CR,FE,NI:*;2) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&FE,CR,FE,NI:*;2) 273.00 +1e-8; 6000.00  N
Ref:14 !

$NI

PARAMETER MQ(BCC_A2&NI,CR:*) 273.00 -407000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&NI,CR:*) 273.00 -17.2*T; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&NI,FE:*) 273.00 -204000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&NI,FE:*) 273.00 +R*T*LN(1.8E-5); 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&NI,NI:*) 273.00 -204000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&NI,NI:*) 273.00 +R*T*LN(1.8E-5); 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&NI,CR,FE:*;0) 273.00 +88000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&NI,CR,FE:*;0) 273.00 +10*T; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&NI,CR,NI:*;0) 273.00 +350000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&NI,CR,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&NI,FE,NI:*;0) 273.00 +150000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&NI,FE,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&NI,CR,FE,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&NI,CR,FE,NI:*;0) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&NI,CR,FE,NI:*;1) 273.00 -500000; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&NI,CR,FE,NI:*;1) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MQ(BCC_A2&NI,CR,FE,NI:*;2) 273.00 +1e-8; 6000.00  N
Ref:14 !
PARAMETER MF(BCC_A2&NI,CR,FE,NI:*;2) 273.00 +1e-8; 6000.00  N
Ref:14 !

"""

ALMGSI_DB = """


$ AL-MG-SI system with metastable phases
$
$ Parameters for metastable phases and mobility
$ taken from E. Povoden-Karadeniz et al, CALPHAD 43 (2011) p. 94
$ 

$Element     Standard state   mass [g/mol]     H_298             S_298
ELEMENT VA   VACUUM            0.0             0.00              0.00       !
ELEMENT AL   FCC_A1           26.98154         4540              28.30      !
ELEMENT MG   HCP_A3           24.305           4998.0            32.671     !
ELEMENT SI   DIA_A4           28.0855          3217.             18.81      !


$
$FUNCTIONS FOR PURE ELEMENT
$
FUNCTION GHSERAL
 273.00 -7976.15+137.093038*T-24.3671976*T*LN(T)
 -1.884662E-3*T**2-0.877664E-6*T**3+74092*T**(-1); 700.00  Y
 -11276.24+223.048446*T-38.5844296*T*LN(T)
 +18.531982E-3*T**2-5.764227E-6*T**3+74092*T**(-1); 933.47  Y
 -11278.378+188.684153*T-31.748192*T*LN(T)-1.231E+28*T**(-9); 2900.00  N
REF:0 !
FUNCTION GHSERMG
 273.00 -8367.34+143.675547*T-26.1849782*T*LN(T)+0.4858E-3*T**2
 -1.393669E-6*T**3+78950*T**(-1); 923.00  Y
 -14130.185+204.716215*T-34.3088*T*LN(T)+1038.192E25*T**(-9); 3000.00  N
REF:0 !
FUNCTION GHSERSI
 273.00 -8162.609+137.236859*T-22.8317533*T*LN(T)
 -1.912904E-3*T**2-0.003552E-6*T**3+176667*T**(-1); 1687.00  Y
 -9457.642+167.271767*T-27.196*T*LN(T)-4.2037E+30*T**(-9); 3600.00  N
REF:0 !

$
$ OTHER FUNCTIONS
$
FUNCTION R         
 273.00 +8.31451; 6000.00  N !
FUNCTION GMG2SI 273.00 -92250.0+440.4*T-75.9*T*LN(T)
   -0.0018*T**2+630000*T**(-1); 6000.00  N
REF:31 !

$
$                                                                       LIQUID
$
 TYPE-DEF % SEQ * !
 PHASE LIQUID % 1  1.0 > 
Random substitutional model. 
>> 6 !
    CONSTITUENT LIQUID  : AL,MG,SI: !

PARAMETER G(LIQUID,AL;0)
 273.00 +11005.029-11.841867*T+7.934E-20*T**7+GHSERAL#; 700.00  Y
 +11005.03-11.841867*T+7.9337E-20*T**7+GHSERAL#; 933.47  Y
 +10482.382-11.253974*T+1.231E+28*T**(-9)+GHSERAL#; 2900.00  N
REF:0 !
PARAMETER G(LIQUID,MG;0)
 273.00 +8202.243-8.83693*T+GHSERMG#-8.0176E-20*T**7; 923.00  Y
 +8690.316-9.392158*T+GHSERMG#-1.038192E+28*T**(-9); 6000.00  N
REF:31 !
PARAMETER G(LIQUID,SI;0)
 273.00 +50696.36-30.099439*T+2.0931E-21*T**7+GHSERSI#; 1687.00  Y
 +49828.165-29.559068*T+4.2037E+30*T**(-9)+GHSERSI#; 3600.00  N
REF:0 !

PARAMETER L(LIQUID,AL,MG;0) 273.00 -12000.0+8.566*T; 6000.00  N
REF:31 !
PARAMETER L(LIQUID,AL,MG;1) 273.00 +1894.0-3.000*T; 6000.00  N
REF:31 !
PARAMETER L(LIQUID,AL,MG;2) 273.00 +2000.0; 6000.00  N
REF:31 !
PARAMETER L(LIQUID,AL,SI;0) 273.00 -11655.93-0.92934*T; 6000.00  N
REF:37 !
PARAMETER L(LIQUID,AL,SI;1) 273.00 -2873.45+0.2945*T; 6000.00  N
REF:37 !
PARAMETER L(LIQUID,AL,SI;2) 273.00 +2520; 6000.00  N
REF:37 !
PARAMETER L(LIQUID,MG,SI;0) 273.00 -70055+24.98*T; 6000.00  N
REF:39 !
PARAMETER L(LIQUID,MG,SI;1) 273.00 -1300; 6000.00  N
REF:39 !
PARAMETER L(LIQUID,MG,SI;2) 273.00 +6272; 6000.00  N
REF:39 !

PARAMETER L(LIQUID,AL,MG,SI;0) 273.00 +11882; 6000.00  N
REF:34 !
PARAMETER L(LIQUID,AL,MG,SI;1) 273.00 -24207; 6000.00  N
REF:34 !
PARAMETER L(LIQUID,AL,MG,SI;2) 273.00 -38223; 6000.00  N
REF:34 !

$
$                                                                       FCC_A1
$
 PHASE FCC_A1  %  2 1   1 > Al-Matrix phase, face-centered cubic >> 6 !
    CONSTITUENT FCC_A1  : AL%,MG,SI : VA :  !

PARAMETER G(FCC_A1,AL:VA;0) 273.00 +GHSERAL#; 2900.00  N
REF:0 !
PARAMETER G(FCC_A1,MG:VA;0) 273.00 +2600-0.90*T+GHSERMG#; 6000.00  N
REF:0 !
PARAMETER G(FCC_A1,SI:VA;0) 273.00 +51000-21.8*T+GHSERSI#; 3600.00  N
REF:0 !

PARAMETER L(FCC_A1,AL,MG:VA;0) 273.00 +1*LDF0ALMG#; 6000.00  N
REF:41 !
PARAMETER L(FCC_A1,AL,MG:VA;1) 273.00 +1*LDF1ALMG#; 6000.00  N
REF:41 !
PARAMETER L(FCC_A1,AL,MG:VA;2) 273.00 +1*LDF2ALMG#; 6000.00  N
REF:41 !
PARAMETER L(FCC_A1,AL,SI:VA;0) 273.00 +1*LDF0ALSI#; 6000.00  N
REF:41 !
PARAMETER L(FCC_A1,AL,SI:VA;1) 273.00 +1*LDF1ALSI#; 6000.00  N
REF:37 !
PARAMETER L(FCC_A1,AL,SI:VA;2) 273.00 +1*LDF2ALSI#; 6000.00  N
REF:37 !
PARAMETER L(FCC_A1,MG,SI:VA;0) 273.00 +1*LDF0SIMG#; 6000.00  N
REF:41 !
PARAMETER L(FCC_A1,MG,SI:VA;1) 273.00 +1*LDF1SIMG#; 6000.00  N
REF:31 !
PARAMETER L(FCC_A1,MG,SI:VA;2) 273.00 +1*LDF2SIMG#; 6000.00  N
REF:31 !

$
$                                                                       SI_DIAMOND_A4
$
 PHASE SI_DIAMOND_A4 % 1 1 > 
Silicon precipitate. Space group Fd3m, prototype: C(diamond) 
>> 4 !
    CONSTITUENT SI_DIAMOND_A4  : AL,MG,SI% : !
PARAMETER G(SI_DIAMOND_A4,AL;0) 273.00 +30.0*T+GHSERAL#; 6000.00  N
REF:31 !
PARAMETER G(SI_DIAMOND_A4,MG;0) 273.00 +GHSERMG#; 6000.00  N
REF:41 !
PARAMETER G(SI_DIAMOND_A4,SI;0) 273.00 +GHSERSI#; 6000.00  N
REF:31 !
PARAMETER L(SI_DIAMOND_A4,AL,SI;0) 273.00 +111417.7-46.1392*T; 6000.00  N
REF:37 !
PARAMETER L(SI_DIAMOND_A4,MG,SI;0) 273.00 +65000; 6000.00  N
REF:41 !

$
$                                                                       MG2SI_B
$
 PHASE MG2SI_B % 2 2 1 > 
Face-centered cubic equilibrium phase. 
Incoherent precipitates (plates or cubes) in the overaging regime, 6xxx alloys. [REF:C31,C32] 
>> 5 !
    CONSTITUENT MG2SI_B  : MG : SI : !
PARAMETER G(MG2SI_B,MG:SI;0) 273.00 GMG2SI; 6000.00  N
REF:31 !

$
$                                                                       BETA_AL3MG2
$
 PHASE BETA_AL3MG2  % 2  89  140 > 
Cubic (Al,Zn)3Mg2 equilibrium phase, prototype: Al3Mg2.  
>> 2 !
    CONSTITUENT BETA_AL3MG2  : MG : AL : !
PARAMETER G(BETA_AL3MG2,MG:AL;0) 273.00 -246175.0-675.5500*T
   +89*GHSERMG#+140*GHSERAL#; 6000.00  N
REF:31 !

$
$                                                                       E_AL30MG23
$
 PHASE E_AL30MG23  %  2  23   30 > 
Epsilon equilibrium phase, prototype: Co5Cr2Mo3 
>> 1 !
    CONSTITUENT E_AL30MG23  : MG : AL :  !
PARAMETER G(E_AL30MG23,MG:AL;0) 273.00 -52565.4-173.1775*T
   +23*GHSERMG#+30*GHSERAL#; 6000.00  N
REF:31 !

$
$                                                                       G_AL12MG17
$
 PHASE G_AL12MG17  %  3  10  24 24 > 
Gamma equilibrium phase, space group: I43m, prototype: Alpha-Mn. 
Precipitate in Al-Mg-Zn alloy 
>> 2 !
    CONSTITUENT G_AL12MG17  : MG : AL,MG% : AL : !
PARAMETER G(G_AL12MG17,MG:MG:MG;0) 273.00 +266939.2-174.638*T+58*GHSERMG#; 6000.00  N
REF:40 !
PARAMETER G(G_AL12MG17,MG:AL:AL;0) 273.00 +195750-203*T
   +10*GHSERMG#+48*GHSERAL#; 6000.00  N
REF:40 !
PARAMETER G(G_AL12MG17,MG:MG:AL;0) 273.00 -105560-101.5*T
   +34*GHSERMG#+24*GHSERAL#; 6000.00  N
REF:40 !
PARAMETER G(G_AL12MG17,MG:AL:MG;0) 273.00 +568249.2-276.138*T
   +34*GHSERMG#+24*GHSERAL#; 6000.00  N
REF:40 !
PARAMETER L(G_AL12MG17,MG:AL:AL,MG;0) 273.00 +226200-29*T; 6000.00  N
REF:40 !
PARAMETER L(G_AL12MG17,MG:MG:AL,MG;0) 273.00 +226200-29*T; 6000.00  N
REF:40 !

$
$                                                                       B_PRIME_L
$
 PHASE B_PRIME_L % 3 3  9  7 > 
Metastable B´ phase - low-T type reflecting lowest energy modification from 1st principles. 
Structure can be related to the hexagonal structure of Q-Phase, with empty Cu-sites. [REF:C11,C37] 
>> 2 !
    CONSTITUENT B_PRIME_L  : AL : MG : SI : !
PARAMETER G(B_PRIME_L,AL:MG:SI;0)  273.00 -140000-10*T+3*GHSERAL#+9*GHSERMG#+7*GHSERSI#; 6000.00  N
REF:41 !

$
$                                                                       MGSI_B_P
$
 PHASE MGSI_B_P % 2 1.8  1 > 
Beta´, metastable hexagonal close-packed rod-like Mg-Si precipitates in 6xxx alloys. [REF:C31,C32,C34] 
>> 5 !
    CONSTITUENT MGSI_B_P  : MG : SI : !

PARAMETER G(MGSI_B_P,MG:SI;0) 273.00 GMG2SI + 24250 - 40.4*T + 5.9*T*LN(T) - 0.0042*T**2 - 130000*T**(-1); 6000.00 N
REF:41 !

$
$                                                                       MG5SI6_B_DP
$
 PHASE MG5SI6_B_DP % 2 5  6 > 
Main metastable hardening phase in 6xxx, monoclinic with space group C2/m. Al-free Mg5Si6.
Semicoherent needles. [REF:C31,C32,C35].     
>> 5 !
    CONSTITUENT MG5SI6_B_DP  : MG : SI : !
PARAMETER G(MG5SI6_B_DP,MG:SI;0)  273.00 -5000-30*T-0.0096*T**2
  -1e-7*T**3+5*GHSERMG#+6*GHSERSI#; 6000.00  N
REF:41 !

$
$                                                                       U1_PHASE
$
 PHASE U1_PHASE % 3 2  1  2 > 
Needle-like precipitate with trigonal structure observed in 6xxx in the beta´ precipitation regime. [REF:C37] 
>> 3 !
    CONSTITUENT U1_PHASE  : AL : MG : SI : !
PARAMETER G(U1_PHASE,AL:MG:SI;0)  273.00 -5000-10*T-0.0055*T**2
  +3e-6*T**3+150000*T**(-1)+2*GHSERAL#+GHSERMG#+2*GHSERSI#; 6000.00  N
REF:41 !

$
$                                                                       U2_PHASE
$
 PHASE U2_PHASE % 3 1  1  1 > 
Needle-like precipitate with orthorhombic structure in 6xxx in the beta' precipitation regime. [REF:C37] 
>> 3 !
    CONSTITUENT U2_PHASE  : AL : MG : SI : !
PARAMETER G(U2_PHASE,AL:MG:SI;0)  273.00 -14000-3.75*T-0.0015*T**2
  +7.5e-7*T**3+62500*T**(-1)+1*GHSERAL#+1*GHSERMG#+1*GHSERSI#; 6000.00  N
REF:41 !

$
$                                                                       Mobility terms
$
PARAMETER MQ(FCC_A1&AL,*) 273.00 -127200+R*T*LN(1.39e-5); 6000.00  N       
Ref:41 !

PARAMETER MQ(FCC_A1&MG,AL:*) 273.00 -119000+R*T*LN(3.7e-5); 6000.00  N
Ref:41!
PARAMETER MQ(FCC_A1&MG,MG:*) 273.00 -112499+R*T*LN(5.7e-5); 6000.00  N
Ref:41!
PARAMETER MQ(FCC_A1&MG,MG,AL:*) 273.00 54511; 6000.00  N
Ref:41!
PARAMETER MQ(FCC_A1&MG,SI:*) 273.00 -119000+R*T*LN(3.7e-5); 6000.00  N
Ref:41!

PARAMETER MQ(FCC_A1&SI,AL:*) 273.00 -136400+R*T*LN(2.31e-4); 6000.00  N
Ref:41 !
PARAMETER MQ(FCC_A1&SI,MG:*) 273.00 -136400+R*T*LN(2.31e-4); 6000.00  N
Ref:41 !
PARAMETER MQ(FCC_A1&SI,SI:*) 273.00 -448400+R*T*LN(154e-4); 6000.00  N
Ref:41 !

$
$ References
$

LIST_OF_REFERENCES

A00201-0    unary                A.T. Dinsdale, SGTE Data of pure elements, CALPHAD, Vol. 15, No. 4, pp 317-425, 1991.
       31   bin, tern            N. Saunders, COST 507: Thermochemical database for light metal alloys, Vol. 2, pp 23-27, 1998.
A00166-34   Al-Mg-Si             J. Lacaze and R. Valdes, CALPHAD-type assessment of the Al-Mg-Si system, Monatshefte f. Chemie, Vol. 136, A00169-37   Al-Fe-Si             Z.-K. Liu, Y.A. Chang, Thermodynamic assessment of the Al-Fe-Si system, Metall. Mater. Trans A, Vol. 30A, pp 1081-1095, 1999. 
A00172-39   Mg-Si-Li             D. Kevorkov, R. Schmid-Fetzer, F. Zhang, Phase equilibria and thermodynamics of the Mg-Si-Li system and remodeling of the Mg-Si system, J. Phase Equilib. Diffusion, Vol. 25, pp 140-151, 2004.
A00173-40   Al-Mg-Zn             P. Liang et al., Experimental investigation and thermodynamic calculation of the Al-Mg-Zn system, Thermochim. Acta, Vol. 314, pp 87-110, 1998.
       41                        E. Povoden-Karadeniz et al, Calphad modeling of metastable phases in the Al-Mg-Si system, CALPHAD, Vol. 42 pp 94-104, 1991
$ ################################################################################################################################################################################################################################################################################

$ References of phase descriptions

       C31 Al-Mg-Si              J.P. Lynch, L.M. Brown, M.H.Jacobs, Microanalysis of age-hardening precipitates in Aluminium-alloys, Acta metall. mater. 30 (1982) 1389.
C00025-C32 Al-Mg-Si              G.A. Edwards, K. Stiller, G.L. Dunlop, M.J. Couper, The precipitaton sequence in Al-Mg-Si alloys, Acta mater. 46 (1998) 3893-3904.
C00026-C33 Al-Mg-Si, GP-zones    K. Matsuda, H. Gamada, K. Fujii, Y. Uetani, T. Sato, A. Kamio, S. Ikeno, High-resolution electron microscopy on the structure of Guinier-Preston zones in an Al-1.6mass% Mg2Si alloy, Metall. mater. trans. A 29 (1998) 1161-1167.
C00027-C34 Mg-Si beta´           R. Vissers, M.A. van Huis, J. Jansen, H.W. Zandbergen, C.D. Marioara, S.J. Andersen, The structure of the beta´ phase in Al-Mg-Si alloys, Acta mater. 55 (2007) 3815-3823.
C00028-C35 Mg-Si beta´´          H.W. Zandbergen, S.J. Andersen, J. Jansen, Structure determination of Mg5Si6 particles in Al by dynamic electron diffraction studies, Science 277 (1997) 1221-1225.
       C36 Al-Mg-Si              H.S. Hasting, A.G. Froseth, S.J. Andersen, R. Vissers, J.C. Walmsley, C.D. Marioara, F. Danoix, W. Lefebvre, R. Holmestad, Composition of beta´´ precipitates in Al-Mg-Si alloys by atom probe tomography and first principles calculations, J. appl. phys. 106 (2009) 123527.
C00029-C37 Al-Mg-Si, U-phases    S.J. Andersen, C.D. Marioara, R. Vissers, A. Froseth, H.W. Zandbergen, The structural relation between precipitates in Al-Mg-Si alloys, the Al-matrix and diamond silicon, with emphasis on the trigonal U1-MgAl2Si2, Mater. sci. eng. A 444 (2007) 157-169.

      11  Smithells Metals Reference Book, Seventh Edition, Butterworth-Heinemann, Oxford, 1999.  
      32  J. Yao, Y.W. Cui, H. Liu, H. Kou, J. Li, L. Zhou, Computer Coupling of Phase Diagrams and Thermochemistry Vol.32, 602-607, 2008.
"""