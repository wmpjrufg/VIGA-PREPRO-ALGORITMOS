################################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR,                  ENG. CIVIL / PROF (UFCAT)
# SYLVIA REGINA MESQUISTA DE ALMEIDA,                 ENG. CIVIL / PROF (UFG-GO)
# MATHEUS HENRIQUE MORATO DE MORAES,                   ENG. CIVIL / PROF (UFCAT)
# GERALDO MAGELA FILHO,                                ENG. CIVIL / PROF (UFCAT)
# GUSTAVO GONÇALVES COSTA,                                    ENG. CIVIL (UFCAT)
################################################################################

################################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIOTECA DE DIMENSIONAMENTO DE VIGAS PRÉ-FABRICADAS E PROTENDIDAS DESENVOL-
# VIDA PELO GRUPO DE PESQUISA E ESTUDOS EM ENGENHARIA (GPEE)
################################################################################

################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import numpy as np

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE

def PROP_GEOMETRICA_I(H, B_FS, B_FI, B_W, H_S, H_I, H_SI, H_II):
    """
    Esta função determina as propriedades geométricas de uma seção I com abas inclinadas.

    Entrada:
    H         | Altura da viga                                     | m    | float
    B_FS      | Base de mesa superior da viga                      | m    | float
    B_FI      | Base de mesa inferior da viga                      | m    | float
    B_W       | Base de alma da viga                               | m    | float
    H_S       | Altura de mesa superior da viga                    | m    | float
    H_I       | Altura de mesa inferior da viga                    | m    | float
    H_SI      | Altura inclinada de mesa superior da viga          | m    | float
    H_II      | Altura inclinada de mesa inferior da viga          | m    | float

    Saída:
    A_C       | Área da  seção transversal da viga                 | m²   | float
    I_C       | Inércia da viga                                    | m^4  | float
    Y_SUP     | Ordenada da fibra superior                         | m    | float 
    Y_INF     | Ordenada da fibra inferior                         | m    | float
    W_SUP     | Modulo de resistência superior                     | m³   | float
    W_INF     | Modulo de resistência inferior                     | m³   | float
    """
    A_1 = B_W * H
    A_2 = (B_FS - B_W) * H_S
    A_3 = ((B_FS - B_W) * H_SI) / 2
    A_4 = (B_FI - B_W) * H_I
    A_5 = ((B_FI - B_W) * H_II)/2
    A_C = A_1 + A_2 + A_3 + A_4 + A_5  
    Y_CG = (A_1 * H / 2 + A_2 * (H - H_S / 2) + A_3 * (H - H_S - H_SI / 3) + A_4 * H_I / 2 + A_5 * (H_I + H_II / 3)) /(A_C)
    I_1 = (B_W * H**3) / 12 + A_1 * (H / 2 - Y_CG)**2 
    I_2 = ((B_FS - B_W)* H_S**3) / 12 + A_2 * (H - H_S/2 - Y_CG)**2 
    I_3 = ((B_FS - B_W)* H_SI**3) / 36 + A_3 * (H - H_S - H_SI / 3 - Y_CG)**2 
    I_4 = ((B_FI - B_W)* H_I**3) / 12 + A_4 * (Y_CG - H_I / 2)**2 
    I_5 = ((B_FI - B_W)* H_II**3) / 36 + A_5 * (Y_CG - H_I - H_II / 3)**2 
    I_C = I_1 + I_2 + I_3 + I_4 + I_5
    Y_SUP = H - Y_CG 
    Y_INF = Y_CG
    W_SUP = I_C / Y_SUP 
    W_INF = I_C / Y_INF     
    return A_C, I_C, Y_SUP, Y_INF, W_SUP, W_INF

def PROP_GEOMETRICA_RET(B_W, H):
    """
    Esta função determina as propriedades geométricas de uma seção retangular.

    Entrada:
    B_W       | Largura da viga                        | m    | float 
    H         | Altura da viga                         | m    | float

    Saída:
    A_C       | Área da  seção transversal da viga     | m²   | float
    I_C       | Inércia da viga                        | m^4  | float
    Y_SUP     | Ordenada da fibra superior             | m    | float 
    Y_INF     | Ordenada da fibra inferior             | m    | float
    W_SUP     | Modulo de resistência superior         | m³   | float
    W_INF     | Modulo de resistência inferior         | m³   | float
    """
    A_C = B_W * H 
    I_C = (B_W * H ** 3) / 12
    Y_SUP = H / 2 
    Y_INF = H / 2
    W_SUP = I_C / Y_SUP 
    W_INF = I_C / Y_INF 
    return A_C, I_C, Y_SUP, Y_INF, W_SUP, W_INF

def FATOR_BETA1(TEMPO, CIMENTO):
    """
    Esta função calcula o valor de BETA_1 que representa a função de 
    crescimento da resistência do cimento.

    Entrada:
    TEMPO       | Tempo                                          | dias  | float
    CIMENTO     | Cimento utilizado                              |       | string    
                |   'CP1' - Cimento portland 1                   |       | 
                |   'CP2' - Cimento portland 2                   |       |              
                |   'CP3' - Cimento portland 3                   |       |
                |   'CP4' - Cimento portland 4                   |       | 
                |   'CP5' - Cimento portland 5                   |       | 
    
    Saída:
    BETA_1      | Parâmetro de crescimento da resistência        |       | float   
    """
    if TEMPO < 28 :
        if CIMENTO == 'CP1' or CIMENTO == 'CP2':
          S = 0.25  
        elif CIMENTO == 'CP3' or CIMENTO == 'CP4':
          S = 0.38  
        elif CIMENTO == 'CP5':
          S = 0.20  
        BETA_1 = np.exp(S * (1 - (28 / TEMPO) ** 0.50))
    else :
        BETA_1 = 1
    return BETA_1

def MODULO_ELASTICIDADE_CONCRETO(AGREGADO, F_CK, F_CKJ):
    """
    Esta função calcula os módulos de elasticidade do concreto.  

    Entrada:
    AGREGADO    | Tipo de agragado usado no traço do cimento       |        | string    
                |   'BAS' - Agregado de Basalto                    |        | 
                |   'GRA' - Agregado de Granito                    |        |              
                |   'CAL' - Agregado de Calcário                   |        |
                |   'ARE' - Agregado de Arenito                    |        | 
    F_CK        | Resistência característica à compressão          | kN/m²  | float   
    F_CKJ       | Resistência característica à compressão idade J  | kN/m²  | float
    
    Saída:
    E_CIJ       | Módulo de elasticidade tangente                  | kN/m²  | float
    E_CSJ       | Módulo de elasticidade do secante                | kN/m²  | float   
    """
    # Determinação do módulo tangente E_CI idade T
    if AGREGADO == 'BAS':         
        ALFA_E = 1.2
    elif AGREGADO == 'GRA':         
        ALFA_E = 1.0
    elif AGREGADO == 'CAL':       
        ALFA_E = 0.9
    elif AGREGADO == 'ARE':       
        ALFA_E = 0.7
    F_CK /= 1E3
    if F_CK <= 50:        
        E_CI = ALFA_E * 5600 * np.sqrt(F_CK)
    elif F_CK > 50:   
        E_CI = 21.5 * (10 ** 3) * ALFA_E * (F_CK / 10 + 1.25) ** (1 / 3)
    ALFA_I = 0.8 + 0.2 * F_CK / 80
    if ALFA_I > 1:        
        ALFA_I = 1
    # Determinação do módulo secante E_CS idade T
    E_CS = E_CI * ALFA_I
    if F_CK <= 45 :
        F_CK *= 1E3
        E_CIJ = E_CI * (F_CKJ / F_CK) ** 0.5  
    elif  F_CK > 45 : 
        F_CK *= 1E3
        E_CIJ = E_CI * (F_CKJ / F_CK) ** 0.3  
    E_CSJ = E_CIJ * ALFA_I
    E_CIJ *= 1E3 
    E_CSJ *= 1E3 
    return E_CIJ, E_CSJ

def PROP_MATERIAL(F_CK, TEMPO, CIMENTO, AGREGADO):
    """
    Esta função determina propriedades do concreto em uma idade TEMPO.
    
    Entrada:
    F_CK        | Resistência característica à compressão                | kN/m²  | float   
    TEMPO       | Tempo                                                  | dias   | float
    CIMENTO     | Cimento utilizado                                      |        | string    
                |   'CP1' - Cimento portland 1                           |        | 
                |   'CP2' - Cimento portland 2                           |        |              
                |   'CP3' - Cimento portland 3                           |        |
                |   'CP4' - Cimento portland 4                           |        | 
                |   'CP5' - Cimento portland 5                           |        | 
    AGREGADO    | Tipo de agragado usado no traço do cimento             |        | string    
                |   'BAS' - Agregado de Basalto                          |        | 
                |   'GRA' - Agregado de Granito                          |        |              
                |   'CAL' - Agregado de Calcário                         |        |
                |   'ARE' - Agregado de Arenito                          |        | 
    
    Saída:
    F_CKJ       | Resistência característica à compressão idade J        | kN/m²  | float
    F_CTMJ      | Resistência média caracteristica a tração idade J      | kN/m²  | float
    F_CTKINFJ   | Resistência média caracteristica a tração inf idade J  | kN/m²  | float
    F_CTKSUPJ   | Resistência média caracteristica a tração sup idade J  | kN/m²  | float
    E_CIJ       | Módulo de elasticidade tangente                        | kN/m²  | float
    E_CSJ       | Módulo de elasticidade do secante                      | kN/m²  | float      
    """
    # Propriedades em situação de compressão F_C idade TEMPO em dias
    BETA_1 = FATOR_BETA1(TEMPO, CIMENTO)
    F_CKJ = F_CK * BETA_1
    F_CKJ /= 1E3
    F_CK /= 1E3
    if F_CKJ < 25 :
        F_CKJ = 25
    # Propriedades em situação de tração F_CT idade TEMPO em dias
    if F_CK <= 50:
          F_CTMJ = 0.3 * F_CKJ ** (2/3)
    elif F_CK > 50:
          F_CTMJ = 2.12 * np.log(1 + 0.11 * F_CKJ)
    F_CTMJ *= 1E3
    F_CTKINFJ = 0.7 * F_CTMJ 
    F_CTKSUPJ = 1.3 * F_CTMJ
    # Módulo de elasticidade do concreto
    F_CKJ *= 1E3
    F_CK *= 1E3
    [E_CIJ, E_CSJ] = MODULO_ELASTICIDADE_CONCRETO(AGREGADO, F_CK, F_CKJ)
    return  F_CKJ, F_CTMJ, F_CTKINFJ, F_CTKSUPJ, E_CIJ, E_CSJ 

def TENSAO_INICIAL(TIPO_PROT, TIPO_ACO, F_PK, F_YK):
    """
    Esta função determina a tensão inicial de protensão e a carga inicial de protensão.

    Entrada:
    TIPO_PROT  | Protensão utilizada                                  |       | string    
               |   'PRE' - Peça pré tracionada                        |       | 
               |   'POS' - Peça pós tracionada                        |       |  
    TIPO_ACO   | Tipo de aço                                          |       | string
               |   'RN' - Relaxação normal                            |       |
               |   'RB' - Relaxação baixa                             |       |
    F_PK       | Tensão última característica do aço                  | kN/m² | float
    F_YK       | Tensão de escoamento característica do aço           | kN/m² | float   

    Saída:
    SIGMA_PIT0 | Tensão inicial de protensão                          | kN/m² | float
    
    """
    if TIPO_PROT == 'PRE':
        if TIPO_ACO == 'RN':
            SIGMA_PIT0 = min(0.77 * F_PK, 0.90 * F_YK)
        elif TIPO_ACO == 'RB':
            SIGMA_PIT0 = min(0.77 * F_PK, 0.85 * F_YK)       
    elif TIPO_PROT == 'POS':
        if TIPO_ACO == 'RN':
            SIGMA_PIT0 = min(0.74 * F_PK, 0.87 * F_YK)
        elif TIPO_ACO == 'RB':
            SIGMA_PIT0 = min(0.74 * F_PK, 0.82 * F_YK)
    return SIGMA_PIT0

def COMPRIMENTO_TRANSFERENCIA(PHI_L, F_YK, F_CTKINFJ, ETA_1, ETA_2, SIGMA_PI, H): ##################################
    """
    Esta função calcula o comprimento de tranferência da armadura L_P.

    Entrada:
    PHI_L      | Diâmetro da armadura                                 | m      | float
    F_YK       | Tensão de escoamento característica do aço           | kN/m²  | float
    F_CTKINFJ  |                                                      |        | float
    ETA_1      |                                                      |        | float
    ETA_2      |                                                      |        | float
    SIGMA_PI   |                                                      |        | float
    H

    Saída:
    L_P 
    """ 
    F_YD = F_YK / 1.15
    F_CTD = F_CTKINFJ / 1.4
    F_BPD = ETA_1 * ETA_2 * F_CTD
    # Comprimento de ancoragem básico para cordoalhas
    L_BP = (7 * PHI_L * F_YD) / (36 * F_BPD)
    # Comprimento básico de transferência para cordoalhas não gradual
    L_BPT = (0.625 * L_BP ) * (SIGMA_PI/ F_YD)
    AUXL_P = np.sqrt(H ** 2 + (0.6 * L_BPT) ** 2) 
    L_P = max(AUXL_P, L_BPT)
    return L_P

def ESFORCOS(Q, L, L_P):
    """
    Esta função determina os esforços atuantes na viga biapoiada.
    
    Entrada:
    Q           | Carga lineramente distribuida      | kN/m  | float
    L           | Comprimento da viga                | m     | float
    L_P         | Comprimento de transferência       | m     | float

    Saída:
    M          | Momento atuante no meio da viga    | kNm    | float
    M_AP       | Momento atuante no L_P             | kNm    | float
    V          | Cortante atuante no apoio da viga  | kN     | float
    """
    # Momento no meio do vão
    M_MV = Q * (L ** 2) / 8
    # Momento no apoio nas condições iniciais e finais
    M_AP = (Q * L / 2) * L_P - (Q * L_P / 1) * (L_P / 2)
    # Cortanto nos apoios
    V_AP = Q * L / 2 
    return M_MV, M_AP, V_AP 

def ESFORCOS_TRANSITORIOS(Q, L, CHI):
    """
    Esta função determina os esforços atuantes na viga biapoiada considerando
    as situações de içamento e armazenamento.
    
    Entrada:
    Q           | Carga lineramente distribuida                               | kN/m  | float
    L           | Comprimento da viga                                         | m     | float
    CHI         | Proporção de L para a posição dos dispositivos de içamento  |       | float

    Saída:
    M_POS       | Momento positivo atuante no meio da viga                    | kNm   | float
    M_NEG       | Momento negativo atuante no apoio da viga                   | kNm   | float
    """
    # Momento no meio do vão
    M_POS = (Q * (L ** 2) / 8) * (1 - 4 * CHI)
    M_NEG = Q * ((CHI * L) ** 2) / 2
    return M_POS, M_NEG

def TENSOES_NORMAIS(P_I, A_C, E_P, W_INF, W_SUP, DELTA_P, DELTA_G1, DELTA_G2, DELTA_G3, DELTA_Q1, DELTA_Q2, PSI_Q1, M_G1, M_G2, M_G3, M_Q1, M_Q2):
    """
    Esta função determina a tensão normal nos bordos inferior e superior da peça.
    
    Entrada:
    P_I         | Carga de protensão considerando as perdas         | kN      | float
    A_C         | Área da  seção transversal da viga                | m²      | float
    E_P         | Excentricidade de protensão                       | m       | float 
    W_SUP       | Modulo de resistência superior                    | m³      | float
    W_INF       | Modulo de resistência inferior                    | m³      | float
    DELTA_      | Coeficientes parciais de segurança (G,Q,P)        |         | float
    PSI_Q1      | Coeficiente parcial de segurança carga Q_1        |         | float
    M_          | Momentos caracteristicos da peça (G,Q)            | kNm     | float  
        
    Saída:
    SIGMA_INF   | Tensão normal fibra inferior                      | kN/m²   | float
    SIGMA_SUP   | Tensão normal fibra superior                      | kN/m²   | float
    """
    # Tensão normal fibras inferiores
    # Parcela da protensão
    AUX_PINF =  DELTA_P * (P_I / A_C + P_I * E_P / W_INF) 
    # Parcela da carga permanente de PP
    AUX_G1INF = -1 * DELTA_G1 * M_G1 / W_INF 
    # Parcela da carga permanente da capa
    AUX_G2INF = -1 * DELTA_G2 * M_G2 / W_INF
    # Parcela da carga permanente do revestimento
    AUX_G3INF = -1 * DELTA_G3 * M_G3 / W_INF
    # Parcela da carga acidental de utilização
    AUX_Q1INF = -1 * DELTA_Q1 * PSI_Q1 * M_Q1 / W_INF
    # Parcela da carga acidental de montagem da peça
    AUX_Q2INF = -1 * DELTA_Q2 * M_Q2 / W_INF
    # Total para parte inferior
    SIGMA_INF = AUX_PINF + (AUX_G1INF + AUX_G2INF + AUX_G3INF ) + (AUX_Q1INF + AUX_Q2INF)
    # Tensão normal fibras Superior
    # Parcela da protensão
    AUX_PSUP =  DELTA_P * (P_I / A_C - P_I * E_P / W_SUP) 
    # Parcela da carga permanente de PP
    AUX_G1SUP = 1 * DELTA_G1 * M_G1 / W_SUP 
    # Parcela da carga permanente da capa
    AUX_G2SUP = 1 * DELTA_G2 * M_G2 / W_SUP
    # Parcela da carga permanente do revestimento
    AUX_G3SUP = 1 * DELTA_G3 * M_G3 / W_SUP
    # Parcela da carga acidental de utilização
    AUX_Q1SUP = 1 * DELTA_Q1 * PSI_Q1 * M_Q1 / W_SUP
    # Parcela da carga acidental de montagem da peça
    AUX_Q2SUP = 1 * DELTA_Q2 * M_Q2 / W_SUP
    # Total para parte inferior
    SIGMA_SUP = AUX_PSUP + (AUX_G1SUP + AUX_G2SUP + AUX_G3SUP) + (AUX_Q1SUP + AUX_Q2SUP)
    return SIGMA_INF, SIGMA_SUP

def VERIFICA_TENSAO_NORMAL_ATO_PROTENSÃO(SIGMA_INF, SIGMA_SUP, SIGMA_TRACMAX, SIGMA_COMPMAX):
    """
    Esta função verifica a restrição de tensão normal em peças estruturais conforme
    disposto na seção 17.2.4.3.2 da NBR 6118.
    
    Entrada:
    SIGMA_INF       | Tensão normal fibra inferior                      | kN/m²   | float
    SIGMA_SUP       | Tensão normal fibra superior                      | kN/m²   | float
    SIGMA_TRACMAX   | Tensão normal máxima na tração                    | kN/m²   | float
    SIGMA_COMPMAX   | Tensão normal máxima na compressão                | kN/m²   | float

    Saída:
    G_0             | Valor da restrição análise bordo inferior         |         | float
    G_1             | Valor da restrição análise bordo superior         |         | float
    """
    # Análise bordo inferior
    if SIGMA_INF >= 0:
        SIGMA_MAX = SIGMA_COMPMAX
        SIGMA = SIGMA_INF
    else:
        SIGMA_MAX = SIGMA_TRACMAX
        SIGMA = np.abs(SIGMA_INF)
    G_0 = (SIGMA / SIGMA_MAX) - 1 
    # Análise bordo superior
    if SIGMA_SUP >= 0:
        SIGMA_MAX = SIGMA_COMPMAX
        SIGMA = SIGMA_SUP
    else:
        SIGMA_MAX = SIGMA_TRACMAX
        SIGMA = np.abs(SIGMA_SUP)
    G_1 = (SIGMA / SIGMA_MAX) - 1 
    return G_0, G_1   



def AREA_ACO_TRANSVERSAL_MODELO_I(ALPHA, P_I, V_SD, F_CTKINFJ, B_W, D, TIPO_CONCRETO, W_INF, A_C, E_P, M_SDMAX, F_CTMJ, F_YWK):
    """
    Esta função verifica o valor da área de aço necessária para a peça de concreto.

    Entrada:
    ALPHA          | Inclinação do cabo protendido                          | graus | float
    P_I            | Carga de protensão considerando as perdas              | kN    | float    
    V_SD           | Cortante de cálculo                                    | kN    | float
    F_CTKINFJ      | Resistência média caracteristica a tração inf idade J  | kN/m² | float
    B_W            | Largura da viga                                        | m     | float  
    D              | Altura útil da seção                                   | m     | float
    TIPO_CONCRETO  | Defina se é concreto protendido ou armado              |       | string
                   |       'CP' - Concreto protendido                       |       |
                   |       'CA' - Concreto armado                           |       |
    W_INF          | Modulo de resistência inferior                         | m³    | float
    A_C            | Área da  seção transversal da viga                     | m²    | float
    E_P            | Excentricidade de protensão                            | m²    | float 
    M_SDMAX        | Momento de cálculo máximo                              | kN.m  | float
    F_CTMJ         | Resistência média caracteristica a tração idade J      | kN/m² | float
    F_YWK          | Resistência característica do aço do estribo           | kN/m² | float

    Saída:
    V_C            | Resitência ao cisalhamento do concreto                 | kN    | float
    V_SW           | Resitência ao cisalhamento da armadura                 | kN    | float    
    A_SW           | Área de aço para cisalhamento                          | m²/m  | float
    """
    # Contribuição do concreto na resistência
    F_CTD = F_CTKINFJ / 1.40
    V_C0 = 0.6 * F_CTD * B_W * D
    if TIPO_CONCRETO == 'CA':
        V_C = V_C0
    elif TIPO_CONCRETO == 'CP':
        # Correção do cisalhamento em função do esforço de protensão P_I
        ALPHA *= np.pi / 180
        N_P = P_I *np.cos(ALPHA)
        V_P = P_I *np.sin(ALPHA)
        V_SD -= 0.90 * V_P 
        AUX = N_P / A_C + (N_P * E_P) / W_INF
        M_0 = 0.90 * W_INF * AUX
        # Cálculo V_C
        V_CCALC = V_C0 * (1 + M_0 / M_SDMAX)
        if V_CCALC > (2 * V_C0):
            V_C = 2 * V_C0
        else:
            V_C = V_CCALC
    # Determinação da armadura
    if V_C >= V_SD:
        V_SW = 0
        F_CTMJ /= 1E3
        F_CTMJ /= 10
        B_W *= 1E2
        F_YWK /= 1E3
        F_YWK /= 10
        A_SW = (20 * F_CTMJ * B_W) / F_YWK
        A_SW /= 1E4
    else:
        ALPHA_EST = np.pi / 2
        V_SW = V_SD - V_C
        F_YWK /= 1E3
        F_YWK /= 10        
        F_YWD = F_YWK / 1.15
        D *= 1E2
        AUX = 0.90 * D * F_YWD * (np.sin(ALPHA_EST) + np.cos(ALPHA_EST))
        A_SW = V_SW / AUX
        A_SW /= 1 / 1E2
        A_SW /= 1E4
    return V_C, V_SW, A_SW 

def RESISTENCIA_BIELA_COMPRIMIDA(F_CK, B_W, D):
    """
    Esta função verifica o valor da resistência da biela comprimida V_RD2.

    Entrada:
    F_CK        | Resistência característica à compressão         | kN/m² | float
    B_W         | Largura da viga                                 | m     | float  
    D           | Altura útil da seção                            | m     | float
    
    Saída:
    V_RD2       | Resitência da biela comprimida                  | kN    | float 
    """
    # Força resistente da biela de compressão
    F_CK /= 1E3 
    ALFA_V2 = (1 - (F_CK / 250))
    F_CK *= 1E3 
    F_CD = F_CK / 1.40
    V_RD2 = 0.27 * ALFA_V2 * F_CD * B_W * D
    return V_RD2

def VERIFICA_BIELA_COMPRIMIDA(V_SD, V_MAX):
    """
    Esta função verifica a restrição do esforço na biela de compressão.
    
    Entrada:
    V_SD       | Cortante de cálculo                               | kN    | float
    V_MAX      | Cortante máximo permitido na biela de compressão  | kN    | float

    Saída:
    G_0        | Valor da restrição analisando o cisalhamento      |       | float
    """
    G_0 = (V_SD / V_MAX) - 1 
    return G_0  

def TENSAO_ACO(E_SCP, EPSILON, EPSILON_P, EPSILON_Y, F_P, F_Y):
    """
    Esta função determina a tensão da armadura de protensão a partir de 
    um valor de deformação.

    Entrada:
    E_SCP       | Módulo de elasticidade do aço protendido        | kN/m² | float
    EPSILON     | Deformação correspondente a tensão SIGMA 
    desejada                                                      |       | float
    EPSILON_P   | Deformação última do aço                        |       | float
    EPSILON_Y   | Deformação escoamento do aço                    |       | float
    F_Y         | Tensão de escoamento do aço                     | kN/m² | float
    F_P         | Tensão última do aço                            | kN/m² | float
    
    Saída:
    SIGMA       | Tensão correspondente a deformação Deformação   | kN/m² | float
    """
    # Determinação da tensão SIGMA correspodente a deformação EPSILON
    if EPSILON < (F_Y / E_SCP) :
        SIGMA = E_SCP * EPSILON
    elif EPSILON >= (F_Y / E_SCP):
        AUX = (F_P - F_Y) / (EPSILON_P - EPSILON_Y)
        SIGMA = F_Y + AUX * (EPSILON - EPSILON_Y)
    return SIGMA

def DEFORMACAO_ACO(E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y):
    """
    Esta função determina a deformação da armadura de protensão a partir de 
    um valor de tensão.

    Entrada:
    E_SCP       | Módulo de elasticidade do aço protendido        | kN/m² | float
    SIGMA       | Tensão correspondente a tensão EPSILON desejada | kN/m² | float
    EPSILON_P   | Deformação última do aço                        |       | float
    EPSILON_Y   | Deformação escoamento do aço                    |       | float
    F_Y         | Tensão de escoamento do aço                     | kN/m² | float
    F_P         | Tensão última do aço                            | kN/m² | float
    
    Saída:
    EPSILON     | Deformação correspondente a tensão SIGMA        |       | float
    """
    # Determinação da deformação EPSILON correspodente a tensão SIGMA
    if (SIGMA / E_SCP) < (F_Y / E_SCP):      
        EPSILON = SIGMA / E_SCP
    elif (SIGMA / E_SCP) >= (F_Y / E_SCP):
        AUX = (F_P - F_Y) / (EPSILON_P - EPSILON_Y)
        EPSILON = (SIGMA - F_Y) / AUX + EPSILON_Y
    return EPSILON

def AREA_ACO_LONGITUDINAL_CP_RET(M_SD, F_CK, B_W, D, E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y):
    """
    Esta função determina a área de aço em elementos de concreto quando submetido a um momento fletor M_SD
    
    TIPO_CONCRETO  | Defina se é concreto protendido ou armado                      |       | string
                   |       'CP' - Concreto protendido                               |       |
                   |       'CA' - Concreto armado                                   |       |
    M_SD           | Momento de cálculo                                             | kN.m  | float
    F_CK           | Resistência característica à compressão                        | kN/m² | float
    B_W            | Largura da viga                                                | m     | float
    D              | Altura útil da seção                                           | m     | float
    E_SCP          | Módulo de elasticidade do aço protendido                       | kN/m² | float
    SIGMA          | Tensão correspondente a tensão EPSILON desejada                | kN/m² | float
    EPSILON_P      | Deformação última do aço                                       |       | float
    EPSILON_Y      | Deformação escoamento do aço                                   |       | float
    F_Y            | Tensão de escoamento do aço                                    | kN/m² | float
    F_P            | Tensão última do aço                                           | kN/m² | float

    Saída:
    X              | Linha neutra da seção medida da parte externa comprimida ao CG | m     | float  
    Z              | Braço de alvanca                                               | m     | float    
    A_S            | Área de aço necessária na seção                                | m²    | float
    EPSILON_S      | Deformação do aço                                              |       | float
    EPSILON_C      | Deformação do concreto                                         |       | float
    """
    # Determinação dos fatores de cálculo de X e A_S
    F_CK /= 1E3
    if F_CK >  50:
        LAMBDA = 0.80 - ((F_CK - 50) / 400)
        ALPHA_C = (1.00 - ((F_CK - 50) / 200)) * 0.85
        EPSILON_C2 = 2.0 + 0.085 * (F_CK - 50) ** 0.53
        EPSILON_C2 = EPSILON_C2 / 1000
        EPSILON_CU = 2.6 + 35.0 * ((90 - F_CK) / 100) ** 4
        EPSILON_CU = EPSILON_CU / 1000
        KX_23 = EPSILON_CU / (EPSILON_CU + 10 / 1000)
        KX_34 = 0.35
    else:
        LAMBDA = 0.80
        ALPHA_C = 0.85
        EPSILON_C2 = 2.0 / 1000
        EPSILON_CU = 3.5 / 1000
        KX_23 = EPSILON_CU / (EPSILON_CU + 10 / 1000)
        KX_34 = 0.45
    # Linhas neutra X
    F_CK *= 1E3
    F_CD = F_CK / 1.40
    PARTE_1 = M_SD / (B_W * ALPHA_C * F_CD)
    NUMERADOR = D - np.sqrt(D ** 2 - 2 * PARTE_1)
    DENOMINADOR = LAMBDA
    X = NUMERADOR / DENOMINADOR
    # Deformações nas fibras comprimidas (concreto) e tracionadas (aço) 
    KX = X / D
    if KX > KX_23:
        EPSILON_C = EPSILON_CU
        EPSILON_S = (1 -  KX) * EPSILON_C 
    elif KX < KX_23:
        EPSILON_S = 10 / 1000
        EPSILON_C = EPSILON_S / (1 - KX)
    elif KX == KX_23:
        EPSILON_S = 10 / 1000
        EPSILON_C = EPSILON_CU 
    # Braço de alavanca Z
    Z = D - 0.50 * LAMBDA * X
    # Área de aço As
    EPSILON_SAUX = DEFORMACAO_ACO(E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y)
    EPSILON_ST = EPSILON_S + EPSILON_SAUX
    F_YD = TENSAO_ACO(E_SCP, EPSILON_ST, EPSILON_P, EPSILON_Y, F_P, F_Y)
    A_S = M_SD / (Z * F_YD)
    return X, EPSILON_S, EPSILON_C, Z, A_S

def AREA_ACO_LONGITUDINAL_CP_T(M_SD, F_CK, B_W, B_F, H_F, D, E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y):
    """
    Esta função determina a área de aço em elementos de concreto quando submetido a um momento fletor M_SD
    
    TIPO_CONCRETO  | Defina se é concreto protendido ou armado                      |       | string
                   |       'CP' - Concreto protendido                               |       |
                   |       'CA' - Concreto armado                                   |       |
    M_SD           | Momento de cálculo                                             | kN.m  | float
    F_CK           | Resistência característica à compressão                        | kN/m² | float
    B_W            | Largura da viga                                                | m     | float
    B_F            | Largura da mesa                                                | m     | float
    H_F            | Altura da mesa                                                 | m     | float
    D              | Altura útil da seção                                           | m     | float
    E_SCP          | Módulo de elasticidade do aço protendido                       | kN/m² | float
    SIGMA          | Tensão correspondente a tensão EPSILON desejada                | kN/m² | float
    EPSILON_P      | Deformação última do aço                                       |       | float
    EPSILON_Y      | Deformação escoamento do aço                                   |       | float
    F_Y            | Tensão de escoamento do aço                                    | kN/m² | float
    F_P            | Tensão última do aço                                           | kN/m² | float

    Saída:
    X              | Linha neutra da seção medida da parte externa comprimida ao CG | m     | float  
    Z              | Braço de alvanca                                               | m     | float    
    A_S            | Área de aço necessária na seção                                | m²    | float
    EPSILON_S      | Deformação do aço                                              |       | float
    EPSILON_C      | Deformação do concreto                                         |       | float
    """
    # Determinação dos fatores de cálculo de X e A_S
    F_CK /= 1E3
    if F_CK >  50:
        LAMBDA = 0.80 - ((F_CK - 50) / 400)
        ALPHA_C = (1.00 - ((F_CK - 50) / 200)) * 0.85
        EPSILON_C2 = 2.0 + 0.085 * (F_CK - 50) ** 0.53
        EPSILON_C2 = EPSILON_C2 / 1000
        EPSILON_CU = 2.6 + 35.0 * ((90 - F_CK) / 100) ** 4
        EPSILON_CU = EPSILON_CU / 1000
        KX_23 = EPSILON_CU / (EPSILON_CU + 10 / 1000)
        KX_34 = 0.35
    else:
        LAMBDA = 0.80
        ALPHA_C = 0.85
        EPSILON_C2 = 2.0 / 1000
        EPSILON_CU = 3.5 / 1000
        KX_23 = EPSILON_CU / (EPSILON_CU + 10 / 1000)
        KX_34 = 0.45
    # Linhas neutra X
    F_CK *= 1E3
    F_CD = F_CK / 1.40
    B_WTESTE = B_F
    PARTE_1 = M_SD / (B_WTESTE * ALPHA_C * F_CD)
    NUMERADOR = D - np.sqrt(D ** 2 - 2 * PARTE_1)
    DENOMINADOR = LAMBDA
    X = NUMERADOR / DENOMINADOR
    if (LAMBDA * X) <= H_F:
        # Deformações nas fibras comprimidas (concreto) e tracionadas (aço) 
        KX = X / D
        if KX > KX_23:
            EPSILON_C = EPSILON_CU
            EPSILON_S = (1 -  KX) * EPSILON_C 
        elif KX < KX_23:
            EPSILON_S = 10 / 1000
            EPSILON_C = EPSILON_S / (1 - KX)
        elif KX == KX_23:
            EPSILON_S = 10 / 1000
            EPSILON_C = EPSILON_CU 
        # Braço de alavanca Z
        Z = D - 0.50 * LAMBDA * X
        # Área de aço As
        EPSILON_SAUX = DEFORMACAO_ACO(E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y)
        EPSILON_ST = EPSILON_S + EPSILON_SAUX
        F_YD = TENSAO_ACO(E_SCP, EPSILON_ST, EPSILON_P, EPSILON_Y, F_P, F_Y)
        A_S = M_SD / (Z * F_YD)
    elif (LAMBDA * X) > H_F:
        # Deformações nas fibras comprimidas (concreto) e tracionadas (aço) 
        KX = X / D
        if KX > KX_23:
            EPSILON_C = EPSILON_CU
            EPSILON_S = (1 -  KX) * EPSILON_C 
        elif KX < KX_23:
            EPSILON_S = 10 / 1000
            EPSILON_C = EPSILON_S / (1 - KX)
        elif KX == KX_23:
            EPSILON_S = 10 / 1000
            EPSILON_C = EPSILON_CU 
        # Braço de alavanca Z
        Z = D - 0.50 * LAMBDA * X
        # Área de aço As
        EPSILON_SAUX = DEFORMACAO_ACO(E_SCP, SIGMA, EPSILON_P, EPSILON_Y, F_P, F_Y)
        EPSILON_ST = EPSILON_S + EPSILON_SAUX
        F_YD = TENSAO_ACO(E_SCP, EPSILON_ST, EPSILON_P, EPSILON_Y, F_P, F_Y)
        M_1SD = (B_F - B_W) * H_F * ALPHA_C * F_CD * (D - 0.50 * H_F)
        M_2SD = M_SD - M_1SD
        A_1S = M_1SD / ((D - 0.50 * H_F) * F_YD)
        A_2S = M_2SD / (Z * F_YD)
        A_S = A_1S + A_2S
    return X, EPSILON_S, EPSILON_C, Z, A_S

def VERIFICA_ARMADURA_FLEXAO(A_SCP, A_SCPNEC):
    """
    Esta função verifica a restrição do esforço na biela de compressão.
    
    Entrada:
    A_SCP      | Armadura de protensão da peça                      | m²    | float
    A_SCPNEC   | Armadura de protensão necessária para peça         | m²    | float

    Saída:
    G_0        | Valor da restrição analisando a armadura de flexão |       | float
    """
    G_0 = (A_SCPNEC / A_SCP) - 1 
    return G_0   

def MOMENTO_MINIMO(W_INF, F_CTKSUPJ):
    """
    Esta função calcula o momento mínimo para gerar a área de aço mínima.

    Entrada:
    W_INF      | Modulo de resistência inferior                         | m³     | float
    F_CTKSUPJ  | Resistência média caracteristica a tração sup idade J  | kN/m²  | float

    Saída:
    M_MIN      | Momento mínimo para armadura mínima                    | kN.m  | float
    """
    M_MIN = 0.80 * W_INF * F_CTKSUPJ
    return M_MIN

def ARMADURA_ASCP_ELS(A_C, I_C, Y_I, E_P, PSI1_Q1, PSI2_Q1, M_G1, M_G2, M_G3, M_Q1, SIGMA_PI, F_CTKINFJ, FATOR_SEC):
    """
    Esta função calcula a área de aço mínima em função dos limites do ELS.

    Entrada:
    A_C         | Área da  seção transversal da viga                       | m²      | float
    I_C         | Inércia da viga                                          | m^4     | float
    Y_I         | Distância do CG que deseja-se calcular a tensão          | m       | float
    E_P         | Excentricidade de protensão                              | m       | float
    PSI1_Q1     | Coeficiente parcial de segurança PSI_1                   |         | float
    PSI2_Q1     | Coeficiente parcial de segurança PSI_2                   |         | float
    M_          | Momentos caracteristicos da peça (G,Q)                   | kN.m    | float
    SIGMA_PI    | Tensão de protensão                                      | kN/m²   | float
    F_CTKINFJ   | Resistência caracteristica a tração inferior na idade j  | kN/m²   | float
    FATOR_SEC   | Fator de correção da resistência                         |         | float

    Saída:
    A_SCPINICIAL| Área de aço inicial respeitando os limites de serviço    | m²      | float
    """
    # ELS-F
    if FATOR_SEC == 'RETANGULAR':
        ALPHA_F = 1.50
    elif FATOR_SEC == 'I':
        ALPHA_F = 1.30
    elif FATOR_SEC == 'DUPLO T':
        ALPHA_F = 1.20
    LIMITE_TRAC0 = - ALPHA_F * F_CTKINFJ
    AUX_0 = SIGMA_PI / A_C + (SIGMA_PI * E_P * Y_I) / I_C
    AUX_1 = ((M_G1 + M_G2 + M_G3) * Y_I) / I_C + ((PSI1_Q1 * M_Q1) * Y_I) / I_C
    A_SCP0 = (LIMITE_TRAC0 +  AUX_1) / AUX_0
    # ELS-D
    LIMITE_TRAC1 = 0
    AUX_2 = ((M_G1 + M_G2 + M_G3) * Y_I) / I_C + ((PSI2_Q1 * M_Q1) * Y_I) / I_C
    A_SCP1 = (LIMITE_TRAC1 + AUX_2) / AUX_0
    A_SCPINICIAL = max(A_SCP0, A_SCP1)
    return A_SCPINICIAL

def ARMADURA_ASCP_ELU(A_C, W_INF, W_SUP, E_P, PSI1_Q1, PSI2_Q1, M_G1, M_G2, M_G3, M_Q1, SIGMA_PI, F_CTMJ, F_CKJ):
    """
    Esta função calcula a área de aço mínima em função dos limites do ELU.

    Entrada:
    A_C         | Área da  seção transversal da viga                       | m²      | float
    W_INF       | Modulo de resistência inferior                           | m³      | float
    W_SUP       | Modulo de resistência superior                           | m³      | float
    E_P         | Excentricidade de protensão                              | m       | float
    PSI1_Q1     | Coeficiente parcial de segurança PSI_1                   |         | float
    PSI2_Q1     | Coeficiente parcial de segurança PSI_2                   |         | float
    M_          | Momentos caracteristicos da peça (G,Q)                   | kN.m    | float
    SIGMA_PI    | Tensão de protensão                                      | kN/m²   | float
    F_CTMJ      | Resistência caracteristica a tração média na idade j     | kN/m²   | float
    F_CKJ       | Resistência característica à compressão idade j          | kN/m²  | float 

    Saída:
    A_SCPINICIAL| Área de aço inicial respeitando os limites do ato de     | m²      | float
                  protensão  
    """
    LIMITE_TRAC0 = -1.20 * F_CTMJ
    AUX_0 = SIGMA_PI / A_C - (SIGMA_PI * E_P) / W_SUP
    AUX_1 = (M_G1 + M_G2 + M_G3) / W_SUP + (PSI1_Q1 * M_Q1) / W_SUP
    A_SCP0 = (LIMITE_TRAC0 -  AUX_1) / AUX_0
    LIMITE_COMP0 = 0.70 * F_CKJ
    AUX_2 = (M_G1 + M_G2 + M_G3) / W_INF + (PSI2_Q1 * M_Q1) / W_INF
    AUX_3 = SIGMA_PI / A_C + (SIGMA_PI * E_P) / W_INF
    A_SCP1 = (LIMITE_COMP0 + AUX_2) / AUX_3
    return A_SCP0, A_SCP1

"""
def ABERTURA_FISSURAS(ALFA_E, P_IINF, A_2, M_SDMAX, D, X_2, I_2, DIAMETRO_ARMADURA, ETA_COEFICIENTE_ADERENCIA, E_SCP, F_CTM, RHO_R) :
    Esta função calcula a abertura de fissuras na peça 

    Entrada:
    ALFA_E      | Relação dos modulos                                    | m²      | float
    W_INF       | Modulo de resistência inferior                         | m³      | float
    PSI1_Q1     | Coeficiente parcial de segurança PSI_1                 |         | float
    PSI2_Q1     | Coeficiente parcial de segurança PSI_2                 |         | float
  
    SIGMA_S = ALFA_E * (P_IINF / A_2) + ( ALFA_E * (M_SDMAX * (D - X_2) / I_2 ) )
    W_1 = (DIAMETRO_ARMADURA / 12.5 * ETA_COEFICIENTE_ADERENCIA) * (SIGMA_S / E_SCP) * 3 (SIGMA_S / F_CTM)
    W_2 = (DIAMETRO_ARMADURA / 12.5 * ETA_COEFICIENTE_ADERENCIA) * (SIGMA_S / E_SCP) * ((4 / RHO_R) + 45)
    W_FISSURA = min(W_1, W_2)
    return W_FISSURA
"""

def GEOMETRIC_PROPERTIES_STATE_I(H, B_F, B_W, H_F, A_SB, ALPHA_MOD, D):
    A_C = (B_F - B_W) * H_F + B_W * H + A_SB * (ALPHA_MOD - 1)
    X_I = ((B_F - B_W) * ((H_F ** 2) / 2) + B_W * ((H ** 2 ) / 2) + A_SB * (ALPHA_MOD - 1) * D) / A_C
    I_I = ((B_F - B_W) * H_F ** 3) / 12 + (B_W * H ** 3) / 12 + (B_F - B_W) * H_F * (X_I - H_F / 2) ** 2 + B_W * H * (X_I - H / 2) ** 2 + A_SB * (ALPHA_MOD - 1) * (X_I - D) ** 2
    return A_C, X_I, I_I

def GEOMETRIC_PROPERTIES_STATE_II(H, B_F, B_W, H_F, A_SB, A_ST, ALPHA_MOD, D, D_L):
    A_1 = B_F / 2
    A_2 = H_F * (0) + (ALPHA_MOD - 1) * A_ST + ALPHA_MOD * A_SB
    A_3 = -D_L*(ALPHA_MOD - 1) * A_ST - D * ALPHA_MOD * A_SB - (H_F ** 2) / 2 * (0)
    X_II = (- A_2 + (A_2 ** 2 - 4 * A_1 * A_3) ** 0.50) / (2 * A_1)
    if X_II <= H_F:
        pass
    elif X_II > H_F:
        A_1 = B_W / 2
        A_2 = H_F * (0) + (ALPHA_MOD - 1) * A_ST + ALPHA_MOD * A_SB
        A_3 = -D_L*(ALPHA_MOD - 1) * A_ST - D * ALPHA_MOD * A_SB - (H_F ** 2) / 2 * (0)
        X_II = (- A_2 + (A_2 ** 2 - 4 * A_1 * A_3) ** 0.50) / (2 * A_1)
    if X_II <= H_F:
        I_II = (B_F * X_II ** 3) / 3 + ALPHA_MOD * A_SB * (X_II - D) ** 2 + (ALPHA_MOD - 1) * A_ST * (X_II - D_L) ** 2
    else:
        I_II = ((B_F - B_W) * H_F ** 3) / 12 + (B_W * X_II **3 ) / 3 + (B_F - B_W) * (X_II - H_F / 2) ** 2 + ALPHA_MOD * A_SB * (X_II - D) ** 2 + (ALPHA_MOD - 1) * A_ST * (X_II - D_L) ** 2
    return X_II, I_II

def BRANSON_INERTIA(M_R, M_D,I_I, I_II):
    M_RMD = (M_R / M_D) ** 3
    I_BRANSON = M_RMD * I_I + (1 - M_RMD) * I_II
    return I_BRANSON

def DISPLACEMENT(BEAM_TYPE, EI, P_K, L):
    if BEAM_TYPE == 0:
        DELTA = 5 * P_K * (L ** 4) * (1 / (384 * EI))
    elif BEAM_TYPE == 1:
        DELTA = 1 * P_K * (L ** 3) * (1 / (48 * EI))
    return DELTA

def AGGREGATE(AGGREGATE_TTYPE):
    if AGGREGATE_TTYPE == 0:
        ALPHA_E = 1.20
    elif AGGREGATE_TTYPE == 1:
        ALPHA_E = 1.00
    elif AGGREGATE_TTYPE == 2:
        ALPHA_E = 0.90
    elif AGGREGATE_TTYPE == 3:
        ALPHA_E = 0.70
    return ALPHA_E

def TANGENT_YOUNG_MODULUS(F_CK, AGGREGATE_TTYPE):
    ALPHA_E = AGGREGATE(AGGREGATE_TTYPE)
    if F_CK >= 20 and F_CK <= 50:
        E_CI = ALPHA_E * 5600 * np.sqrt(F_CK)
    elif F_CK > 50 and F_CK <= 90:
        E_CI = 21.5E3 * ALPHA_E * (F_CK / 10 + 1.25) ** (1 / 3)
    return E_CI

def SECANT_YOUNG_MODULUS(F_CK, E_CI):
    ALPHA_I = 0.80 + 0.20 * (F_CK / 80)
    if ALPHA_I > 1.00:
        ALPHA_I = 1.00
    E_CS = ALPHA_I * E_CI
    return E_CS

def M_R_BENDING_MOMENT(FATOR_SEC, F_CT, H, X_I, I_I, P_I, A_C, W_INF, E_P):
    if FATOR_SEC == 'RETANGULAR':
        ALPHA_F = 1.50
    elif FATOR_SEC == 'I':
        ALPHA_F = 1.30
    elif FATOR_SEC == 'DUPLO T':
        ALPHA_F = 1.20
    Y_T = H - X_I
    AUX = (1 / A_C) + (E_P / W_INF)
    M_0 = P_I * W_INF * AUX
    M_R = M_0 + ALPHA_F * F_CT * (I_I / Y_T)
    return M_R

def EPSILON_COEFFICIENT(T):
    if T < 70:
        EPSILON = 0.68 * (0.996 ** T) * (T ** 0.32)
    else:
        EPSILON = 2
    return EPSILON

def TOTAL_DISPLACEMENT(DELTA_INITIAL, A_ST, B_W, D, T_INITIAL, T_END):
    EPSILON_INITITAL = EPSILON_COEFFICIENT(T_INITIAL)
    EPSILON_END = EPSILON_COEFFICIENT(T_END)
    DELTA_EPSILON = EPSILON_END - EPSILON_INITITAL
    PHO_L = A_ST / (B_W * D)
    ALPHA_F = DELTA_EPSILON / (1 + PHO_L)
    DELTA_TOTAL = DELTA_INITIAL * (1 + ALPHA_F)
    return PHO_L, ALPHA_F, DELTA_TOTAL

def VERIFICA_FLECHA(A_TOTAL, A_MAX):
    """
    Saída:
    G_0        | Valor da restrição analisando o cisalhamento      |       | float
    """
    G_0 = (A_TOTAL / A_MAX) - 1 
    return G_0  