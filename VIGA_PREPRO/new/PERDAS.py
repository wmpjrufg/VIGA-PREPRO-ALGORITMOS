################################################################################
# UNIVERSIDADE FEDERAL DE CATALÃO (UFCAT)
# WANDERLEI MALAQUIAS PEREIRA JUNIOR,                  ENG. CIVIL / PROF (UFCAT)
# MATHEUS HENRIQUE MORATO DE MORAES                    ENG. CIVIL / PROF (UFCAT)
# GUSTAVO GONÇALVES COSTA,                                    ENG. CIVIL (UFCAT)
################################################################################

################################################################################
# DESCRIÇÃO ALGORITMO:
# BIBLIOTECA DE PERDAS DE PROTENSÃO DESENVOLVIDA PELO GRUPO DE PESQUISA E ESTU-
# DOS EM ENGENHARIA (GPEE)
################################################################################


################################################################################
# BIBLIOTECAS NATIVAS PYTHON
import numpy as np

################################################################################
# BIBLIOTECAS DESENVOLVEDORES GPEE

def PERDA_DESLIZAMENTO_ANCORAGEM(P_IT0, SIGMA_PIT0, A_SCP, L_0, DELTA_ANC, E_SCP):
    """
    Esta função determina a perda de protensão por deslizamento da armadura na anco-
    ragem.
    
    Entrada:
    P_IT0       | Carga inicial de protensão                        | kN    | float
    SIGMA_PIT0  | Tensão inicial de protensão                       | kN/m² | float
    A_SCP       | Área de total de armadura protendida              | m²    | float
    L_0         | Comprimento da pista de protensão                 | m     | float
    DELTA_ANC   | Previsão do deslizamento do sistema de ancoragem  | m     | float
    E_SCP       | Módulo de Young do aço protendido                 | kN/m² | float

    Saída:
    DELTAPERC   | Perda percentual de protensão                     | %     | float
    P_IT1       | Carga final de protensão                          | kN    | float
    SIGMA_PIT1  | Tensão inicial de protensão                       | kN/m² | float
    """
    # Pré-alongamento do cabo
    DELTAL_P = L_0 * (SIGMA_PIT0 / E_SCP)
    # Redução da deformação na armadura de protensão
    DELTAEPSILON_P = DELTA_ANC / (L_0 +  DELTAL_P)
    # Perdas de protensão
    DELTASIGMA = E_SCP * DELTAEPSILON_P
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def PERDA_DEFORMACAO_CONCRETO(E_SCP, E_CCP, P_IT0, SIGMA_PIT0, A_SCP, A_C, I_C, E_P, M_GPP):
    """
    Esta função determina a perda de protensão devido a deformação inicial do concreto. 
    
    Entrada:
    E_SCP       | Módulo de Young do aço protendido                 | kN/m² | float
    E_CCP       | Módulo de Young do concreto                       | kN/m² | float
    P_IT0       | Carga inicial de protensão                        | kN    | float
    SIGMA_PIT0  | Tensão inicial de protensão                       | kN/m² | float
    A_C         | Área bruta da seção                               | m²    | float
    A_SCP       | Área de total de armadura protendida              | m²    | float
    I_C         | Inércia da seção bruta                            | m^4   | float
    E_P         | Excentricidade de protensão                       | m     | float 
    M_GPP       | Momento fletor devido ao peso próprio             | kN.m  | float 
      
    Saída:
    DELTAPERC   | Perda percentual de protensão                     | %     | float
    P_IT1       | Carga final de protensão                          | kN    | float
    SIGMA_PIT1  | Tensão inicial de protensão                       | kN/m² | float
    """
    # Perdas de protensão
    ALPHA_P = E_SCP / E_CCP
    AUX_0 = P_IT0 / A_C
    AUX_1 = (P_IT0 * E_P ** 2) / I_C
    AUX_2 = (M_GPP * E_P) / I_C
    DELTASIGMA = ALPHA_P * (AUX_0 + AUX_1 - AUX_2)
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def INTERPOLADOR (X_1, X_2, X_K, Y_1, Y_2):
    """
    Esta função interpola linearmente valores.

    Entrada:
    X_1   | Valor inferior X_K     |       | float
    X_2   | Valor superior X_K     |       | float
    Y_1   | Valor inferior Y_K     |       | float
    Y_2   | Valor superior Y_K     |       | float
    X_K   | Valor X de referência  |       | float

    Saída:
    Y_K   | Valor interpolado Y    |       | float
    """
    Y_K = Y_1 + (X_K - X_1) * ((Y_2 - Y_1) / (X_2 - X_1))
    return Y_K 

def TABELA_PSI1000(TIPO_FIO_CORD_BAR, TIPO_ACO, RHO_SIGMA):
    """
    Esta função encontra o fator Ψ_1000 para cálculo da relaxação.

    Entrada:
    TIPO_FIO_CORD_BAR  | Tipo de armadura de protensão de acordo com a aderência escolhida                 |       | string
                       |    'FIO' - Fio                                                                    |       |
                       |    'COR' - Cordoalha                                                              |       |
                       |    'BAR' - BARRA                                                                  |       |
    TIPO_ACO           | Tipo de aço                                                                       |       | string
                       |    'RN' - Relaxação normal                                                        |       |
                       |    'RB' - Relaxação baixa                                                         |       |
    RHO_SIGMA          | Razão entre F_PK e SIGMA_PI                                                       |       | float

    Saída:
    PSI_1000           | Valor médio da relaxação, medidos após 1.000 h, à temperatura constante de 20 °C  | %     | float     
    """
    # Cordoalhas
    if TIPO_FIO_CORD_BAR == 'COR':
        if TIPO_ACO == 'RN':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 3.50
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 3.50; Y_1 = 7.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 7.00; Y_1 = 12.00
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA 
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)   
        elif TIPO_ACO == 'RB':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 1.30
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 1.30; Y_1 = 2.50
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 2.50; Y_1 = 3.50
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
    # Fio
    elif TIPO_FIO_CORD_BAR == 'FIO':
        if TIPO_ACO == 'RN':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 2.50
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 2.50; Y_1 = 5.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 5.00; Y_1 = 8.50
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA   
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1) 
        elif TIPO_ACO == 'RB':
            if RHO_SIGMA <= 0.5:
                PSI_1000 = 0 
            elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 1.00
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 1.00; Y_1 = 2.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)
            elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 2.00; Y_1 = 3.00
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA  
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)  
    # Barra
    elif TIPO_FIO_CORD_BAR == 'BAR':
        if RHO_SIGMA <= 0.5:
                PSI_1000 = 0 
        elif 0.5 < RHO_SIGMA and RHO_SIGMA <= 0.6:
                Y_0 = 0.00; Y_1 = 1.50
                X_0 = 0.50; X_1 = 0.60
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1) 
        elif 0.6 < RHO_SIGMA and RHO_SIGMA <= 0.7:
                Y_0 = 1.50; Y_1 = 4.00
                X_0 = 0.60; X_1 = 0.70
                X_K = RHO_SIGMA
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1) 
        elif 0.7 < RHO_SIGMA and RHO_SIGMA <= 0.8:
                Y_0 = 4.00; Y_1 = 7.00
                X_0 = 0.70; X_1 = 0.80
                X_K = RHO_SIGMA 
                PSI_1000 = INTERPOLADOR(X_0, X_1, X_K, Y_0, Y_1)        
    return PSI_1000  

def PERDA_RELAXACAO_ARMADURA(P_IT0, SIGMA_PIT0, T_0, T_1, TEMP, F_PK, A_SCP, TIPO_FIO_CORD_BAR, TIPO_ACO):
    """
    Esta função determina a perda de protensão por relaxação da armadura de protensão em peças de concreto 
    protendido.
    
    Entrada:
    P_IT0              | Carga inicial de protensão                                         | kN    | float
    SIGMA_PIT0         | Tensão inicial de protensão                                        | kN/m² | float
    T_0                | Tempo inicial de análise sem correção da temperatura               | dias  | float
    T_1                | Tempo final de análise sem correção da temperatura                 | dias  | float 
    TEMP               | Temperatura de projeto                                             | °C    | float 
    F_PK               | Tensão última do aço                                               | kN/m² | float
    A_SCP              | Área de total de armadura protendida                               | m²    | float
    TIPO_FIO_CORD_BAR  | Tipo de armadura de protensão de acordo com a aderência escolhida  |       | string
                       |    'FIO' - Fio                                                     |       |
                       |    'COR' - Cordoalha                                               |       |
                       |    'BAR' - BARRA                                                   |       |
    TIPO_ACO           | Tipo de aço                                                        |       | string
                       |    'RN' - Relaxação normal                                         |       |
                       |    'RB' - Relaxação baixa                                          |       |
      
    Saída:
    DELTAPERC          | Perda percentual de protensão                                      | %     | float
    P_IT1              | Carga final de protensão                                           | kN    | float
    SIGMA_PIT1         | Tensão inicial de protensão                                        | kN/m² | float
    PSI                | Coeficiente de relaxação do aço                                    | %     | float
    """
    # Determinação PSI_1000
    RHO_SIGMA = SIGMA_PIT0 / F_PK 
    if T_1 > (20 * 365):  
          PSI_1000 = 2.5
    else:
          PSI_1000 = TABELA_PSI1000(TIPO_FIO_CORD_BAR, TIPO_ACO, RHO_SIGMA)         
    # Determinação do PSI no intervalo de tempo T_1 - T_0
    DELTAT_COR = (T_1 - T_0) * TEMP / 20
    PSI =  PSI_1000 * (DELTAT_COR / 41.67) ** 0.15
    # Perdas de protensão
    DELTASIGMA = (PSI / 100) * SIGMA_PIT0
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1, PSI

def CALCULO_HFIC(U, A_C, MU_AR):
    """
    Esta função calcula a altura fictícia de uma peça de concreto.

    Entrada:
    U       | Umidade do ambiente no intervalo de tempo de análise         | %     | float
    A_C     | Área bruta da seção                                          | m²    | float
    MU_AR   | Parte do perímetro externo da seção em contato com ar        | m     | float

    Saída:
    H_FIC   | Altura fictícia da peça para cálculo de fluência e retração  | m     | float
    """
    GAMMA = 1 + np.exp(-7.8 + 0.1 * U)
    H_FIC = GAMMA * 2 * A_C / MU_AR
    if H_FIC > 1.60:
        H_FIC = 1.60
    if H_FIC < 0.050:
        H_FIC = 0.050
    return H_FIC

def BETAS_RETRACAO(T_FIC, H_FIC):
    """
    Esta função determina o coeficiente de retração β_S.

    Entrada:
    T_FIC                  | Tempo de projeto corrigido em função da temperatura          | dias  | float 
    H_FIC                  | Altura fictícia da peça para cálculo de fluência e retração  | m     | float

    Saída:
    BETA_S, A, B, C , D, E | Coeficientes da retração                                     |       | float
    """
    # Coeficientes A até E
    A = 40
    B = 116 * pow(H_FIC, 3) - 282 * pow(H_FIC, 2) + 220 * H_FIC - 4.8
    C = 2.5 * pow(H_FIC, 3) - 8.8 * H_FIC + 40.7
    D = -75 * pow(H_FIC, 3) + 585 * pow(H_FIC, 2) + 496 * H_FIC - 6.8
    E = -169 * pow(H_FIC, 4) + 88 * pow(H_FIC, 3) + 584 * pow(H_FIC, 2) - 39 * H_FIC + 0.8
    T_FIC100 = T_FIC / 100;
    # Determinação do BETA_S
    AUX_0 = pow(T_FIC100, 3) + A * pow(T_FIC100, 2) + B * T_FIC100
    AUX_1 = pow(T_FIC100, 3) + C * pow(T_FIC100, 2) + D * T_FIC100 + E
    BETA_S =  AUX_0 / AUX_1
    return BETA_S, A, B, C, D, E

def BETAF_FLUENCIA(T_FIC, H_FIC):
    """
    Esta função determina o coeficiente de retração β_F.

    Entrada:
    T_FIC                | Tempo de projeto corrigido em função da temperatura          | dias  | float 
    H_FIC                | Altura fictícia da peça para cálculo de fluência e retração  | m     | float

    Saída:
    BETA_F, A, B, C , D  | Coeficientes da fluência                                     |       | float
    """
    A = 42 * pow(H_FIC, 3) - 350 * pow(H_FIC, 2) + 588 * H_FIC + 113
    B = 768 * pow(H_FIC,3) - 3060 * pow(H_FIC, 2) + 3234 * H_FIC - 23
    C = -200 * pow(H_FIC, 3) + 13 * pow(H_FIC, 2) + 1090 * H_FIC + 183
    D = 7579 * pow(H_FIC,3) - 31916 * pow(H_FIC, 2) + 35343 * H_FIC + 1931
    # Determinação do BETA_F
    AUX_0 = pow(T_FIC, 2) + A * T_FIC + B 
    AUX_1 = pow(T_FIC, 2) + C * T_FIC + D
    BETA_F =  AUX_0 / AUX_1
    return BETA_F, A, B, C , D

def CALCULO_TEMPO_FICTICIO(T, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO):
    """
    Esta função calcula o tempo corrigido para cálculo das perdas de fluência e retração. 

    Entrada:
    T                   | Tempo para análise da correção em função da temperatura    | dias  | float
    TEMP                | Temperatura de projeto                                     | °C    | float 
    TIPO_PERDA          | Tipo da perda que deseja-se calcular a correção do tempo   |       | string
                        |       'LENTO'  - Endurecimento lento AF250, AF320, POZ250  |       |
                        |       'NORMAL' - Endurecimento normal CP250, CP320, CP400  |       |
                        |       'RAPIDO' - Endurecimento rápido aderência            |       |
    TIPO_ENDURECIMENTO  | Tipo de enduricmento do cimento                            |       | string
                        |       'RETRACAO' - Retração                                |       |
                        |       'FLUENCIA' - Fluência                                |       |                                                                                           

    Saída:
    T_FIC               | Tempo de projeto corrigido em função da temperatura        | °C    | float 
    """
    # Parâmetros de reologia e tipo de pega
    if TIPO_PERDA == 'RETRACAO':
        ALFA = 1
    elif TIPO_PERDA == 'FLUENCIA':
        if TIPO_ENDURECIMENTO == 'LENTO':
            ALFA = 1
        elif TIPO_ENDURECIMENTO == 'NORMAL':
            ALFA = 2
        elif TIPO_ENDURECIMENTO == 'RAPIDO':
            ALFA = 3
    # Correção dos tempos menores que 3 dias e maiores que 10.000 dias
    if T < 3 and T > 0:
        T = 3
    elif T > 10000:
        T = 10000
    # Determinação da idade fictícia do concreto
    T_FIC = ALFA * ((TEMP + 10) / 30) * T
    return T_FIC 

def PERDA_RETRACAO_CONCRETO(P_IT0, SIGMA_PIT0, A_SCP, U, ABAT, A_C, MU_AR, T_0, T_1, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO, E_SCP):
    """
    Esta função determina a perda de protensão devido a retração do concreto. 
    
    Entrada:
    P_IT0               | Carga inicial de protensão                                 | kN    | float
    SIGMA_PIT0          | Tensão inicial de protensão                                | kN/m² | float
    A_SCP               | Área de total de armadura protendida                       | m²    | float
    U                   | Umidade do ambiente no intervalo de tempo de análise       | %     | float
    ABAT                | Abatimento ou slump test do concreto                       | kN/m² | float
    A_C                 | Área bruta da seção                                        | m²    | float 
    MU_AR               | Parte do perímetro externo da seção em contato com ar      | m     | float
    T_0                 | Tempo inicial de análise sem correção da temperatura       | dias  | float
    T_1                 | Tempo final de análise sem correção da temperatura         | dias  | float 
    TEMP                | Temperatura de projeto                                     | °C    | float 
    TIPO_PERDA          | Tipo da perda que deseja-se calcular a correção do tempo   |       | string
                        |       'LENTO'  - Endurecimento lento AF250, AF320, POZ250  |       |
                        |       'NORMAL' - Endurecimento normal CP250, CP320, CP400  |       |
                        |       'RAPIDO' - Endurecimento rápido aderência            |       |
    TIPO_ENDURECIMENTO  | Tipo de enduricmento do cimento                            |       | string
                        |       'RETRACAO' - Retração                                |       |
                        |       'FLUENCIA' - Fluência                                |       |   
    E_SCP               | Módulo de Young do aço protendido                          | kN/m² | float                                                                                        
      
    Saída:
    DELTAPERC           | Perda percentual de protensão                              | %     | float
    P_IT1               | Carga final de protensão                                   | kN    | float
    SIGMA_PIT1          | Tensão inicial de protensão                                | kN/m² | float
    """
    # Cálculo da defomração específica EPSILON_1S
    EPSILON_1S = -8.09 + (U / 15) - (U ** 2 / 2284) - (U ** 3 / 133765) + (U ** 4 / 7608150)
    EPSILON_1S /= -1E4 
    if U <= 90 and (ABAT >= 0.05 and ABAT <= 0.09):          # intervalo 0.05 <= ABAT <= 0.09
        EPSILON_1S *= 1.00
    elif U <= 90 and (ABAT >= 0.00 and ABAT <= 0.04):        # intervalo 0.00 <= ABAT <= 0.04
        EPSILON_1S *= 0.75
    elif U <= 90 and (ABAT >= 0.10 and ABAT <= 0.15):        # intervalo 10.0 <= ABAT <= 15.0
        EPSILON_1S *= 1.25
    # Cálculo da defomração específica EPSILON_2S
    H_FIC = CALCULO_HFIC(U, A_C, MU_AR)
    EPSILON_2S = (0.33 + 2 * H_FIC) / (0.208 + 3 * H_FIC)
    # Coeficiente BETA_S T0
    T_0FIC = CALCULO_TEMPO_FICTICIO(T_0, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO)
    BETA_S0, _, _, _, _, _, = BETAS_RETRACAO(T_0FIC, H_FIC)
    # Coeficiente BETA_S T1
    T_1FIC = CALCULO_TEMPO_FICTICIO(T_1, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO)
    BETA_S1, _, _, _, _, _, = BETAS_RETRACAO(T_1FIC, H_FIC)
    # Valor final da deformação por retração EPSILON_CS
    EPSILON_CS = EPSILON_1S * EPSILON_2S * (BETA_S1 - BETA_S0)
    # Perdas de protensão
    DELTASIGMA = E_SCP * EPSILON_CS
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def PHI_FLUENCIA(F_CKJ, F_CK, U, A_C, MU_AR, ABAT, T_0, T_1, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO):
    """
    Esta função determina os fatores de fluência φ(t, t_0) de estruturas de concreto.

    Entrada:
    F_CK                | Resistência característica à compressão                    | kN/m² | float   
    F_CKJ               | Resistência característica à compressão idade J            | kN/m² | float
    U                   | Umidade do ambiente no intervalo de tempo de análise       | %     | float
    A_C                 | Área bruta da seção                                        | m²    | float 
    MU_AR               | Parte do perímetro externo da seção em contato com ar      | m     | float
    ABAT                | Abatimento ou slump test do concreto                       | kN/m² | float
    T_0                 | Tempo inicial de análise sem correção da temperatura       | dias  | float
    T_1                 | Tempo final de análise sem correção da temperatura         | dias  | float 
    TEMP                | Temperatura de projeto                                     | °C    | float 
    TIPO_PERDA          | Tipo da perda que deseja-se calcular a correção do tempo   |       | string
                        |       'LENTO'  - Endurecimento lento AF250, AF320, POZ250  |       |
                        |       'NORMAL' - Endurecimento normal CP250, CP320, CP400  |       |
                        |       'RAPIDO' - Endurecimento rápido aderência            |       |
    TIPO_ENDURECIMENTO  | Tipo de enduricmento do cimento                            |       | string
                        |       'RETRACAO' - Retração                                |       |
                        |       'FLUENCIA' - Fluência                                |       |   
    
    Saída:
    PHI                 | Fator de fluência                                          |       | float
    """
    # Fator PHI_A
    F_CK /= 1E3
    if F_CK <= 45:
        F_CK *= 1E3
        PHI_A = 0.80 * (1 - F_CKJ / F_CK)
    elif F_CK > 45: 
        F_CK *= 1E3
        PHI_A = 1.40 * (1 - F_CKJ / F_CK)
    # Fator PHI_1C
    PHI_1C = 4.45 - 0.035 * U
    if U <= 90 and (ABAT >= 0.05 and ABAT <= 0.09):        # intervalo 0.05 <= ABAT <= 0.09
        PHI_1C *= 1.00
    elif U <= 90 and (ABAT >= 0.00 and ABAT <= 0.04):      # intervalo 0.00 <= ABAT <= 0.04
        PHI_1C *= 0.75
    elif U <= 90 and (ABAT >= 10.0 and ABAT <= 15.0):      # intervalo 10.0 <= ABAT <= 15.0
        PHI_1C *= 1.25
    # Fator PHI_2C
    H_FIC = CALCULO_HFIC(U, A_C, MU_AR) 
    PHI_2C = (0.42 + H_FIC) / (0.20 + H_FIC)
    F_CK /= 1E3
    if F_CK <= 45:
        PHI_F = PHI_1C * PHI_2C
    elif F_CK > 45:
        PHI_F = 0.45 * PHI_1C * PHI_2C
    F_CK *= 1E3
    # Fator PHI_D
    PHI_D = 0.40
    # Cálculo fatores BETA
    T_0FIC = CALCULO_TEMPO_FICTICIO(T_0, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO)
    T_1FIC = CALCULO_TEMPO_FICTICIO(T_1, TEMP, TIPO_PERDA, TIPO_ENDURECIMENTO)
    DELTA_T = T_1FIC - T_0FIC
    BETA_D = (DELTA_T + 20) / (DELTA_T + 70)
    BETA_F0, _, _, _, _ = BETAF_FLUENCIA(T_0FIC, H_FIC)
    BETA_F1, _, _, _, _ = BETAF_FLUENCIA(T_1FIC, H_FIC)
    # Coeficiente de fluência
    PHI = PHI_A + PHI_F * (BETA_F1 - BETA_F0) + PHI_D * BETA_D
    return PHI

def PERDA_POR_FLUENCIA_NO_CONCRETO(P_IT0, SIGMA_PIT0, A_SCP, PHI, E_SCP, E_CCP28, SIGMA_CABO):
    """
    Esta função determina a perda de protensão devido ao efeito de fluência.

    Entrada:
    PHI         | Fator de fluência para cada carregamento estudado  |       | Py list
    E_SCP       | Módulo de Young do aço protendido                  | kN/m² | float
    E_CCP28     | Módulo de Young do concreto aos 28 dias            | kN/m² | float
    SIGMA_CABO  | Tensões no nível dos cabos                         | kN/m² | Py list
    P_IT0       | Carga inicial de protensão                         | kN    | float
    SIGMA_PIT0  | Tensão inicial de protensão                        | kN/m² | float
    A_SCP       | Área de total de armadura protendida               | m²    | float

    Saída:
    DELTAPERC   | Perda percentual de protensão                      | %     | float
    P_IT1       | Carga final de protensão                           | kN    | float
    SIGMA_PIT1  | Tensão inicial de protensão                        | kN/m² | float
    """
    TAM = len(PHI)
    # Perdas de protensão
    AUX = 0
    for I_CONT in range(TAM):
        AUX += SIGMA_CABO[I_CONT] * PHI[I_CONT]
    ALPHA_P = E_SCP / E_CCP28
    DELTASIGMA = ALPHA_P * AUX
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1

def INTERACAO_ENTRE_PERDAS_PROGRESSIVAS(E_P, A_C, I_C, E_SCP, E_CCP28, DELTASIGMA_RETRACAO, DELTASIGMA_FLUENCIA, SIGMA_PIT0, PSI, A_SCP, PHI_0):
    """
    Esta função determina a perdas progressivas de protensão considerando o efeito conjunto
    das perdas.

    Entrada:
    A_C                 | Área da  seção transversal da viga         | m²    | float
    I_C                 | Inércia da viga                            | m^4   | float
    E_P                 | Excentricidade de protensão                | m     | float 
    E_SCP               | Módulo de Young do aço protendido          | kN/m² | float
    E_CCP28             | Módulo de Young do concreto idade 28 dias  | kN/m² | float
    DELTASIGMA_RETRACAO | Perda de protensão devido a retração       | kN/m² | float
    DELTASIGMA_FLUENCIA | Perda de protensão devido a fluência       | kN/m² | float
    SIGMA_PIT0          | Tensão inicial de protensão                | kN/m² | float
    PSI                 | Coeficiente de relaxação do aço            | %     | float    
    A_SCP               | Área de total de armadura protendida       | m²    | float
    PHI_0               | Fator de fluência carga P_i e Permanente   |       | float
                        |   no instante t_0                          |       | 
    Saída:
    DELTAPERC           | Perda percentual de protensão              | %     | float
    P_IT1               | Carga final de protensão                   | kN    | float
    SIGMA_PIT1          | Tensão inicial de protensão                | kN/m² | float
    """
    # Perdas de protensão
    PSI /= 1E2
    CHI = - np.log(1 - PSI)
    CHI_P = 1 + CHI
    CHI_C = 1 + 0.50 * PHI_0
    ALPHA_P = E_SCP / E_CCP28
    ETA = 1 + ((E_P ** 2) * (A_C / I_C))
    PHO_P = A_SCP / A_C
    AUX_0 =  DELTASIGMA_RETRACAO + DELTASIGMA_FLUENCIA + SIGMA_PIT0 * PSI
    AUX_1 = CHI_P + CHI_C * ALPHA_P * ETA * PHO_P
    DELTASIGMA = AUX_0 / AUX_1
    SIGMA_PIT1 = SIGMA_PIT0 - DELTASIGMA
    DELTAP = DELTASIGMA * A_SCP
    P_IT1 = P_IT0 - DELTAP
    DELTAPERC = (DELTAP / P_IT0) * 100
    return DELTAPERC, P_IT1, SIGMA_PIT1