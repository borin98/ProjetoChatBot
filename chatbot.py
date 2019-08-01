import numpy as np
import tensorflow as tf
import re
import time

"""
Função que faz a limpeza de alguns caracteres

@:param -> dataframe original
@:return -> texto limpo

"""

def limpaTexto ( df ) :

    df = df.lower (  )

    df = re.sub(r"i'm", "i am", df)
    df = re.sub(r"he's", "he is", df)
    df = re.sub(r"she's", "she is", df)
    df = re.sub(r"that's", "that is", df)
    df = re.sub(r"what's", "what is", df)
    df = re.sub(r"where's", "where is", df)
    df = re.sub(r"won't", "will not", df)
    df = re.sub(r"can't", "cannot", df)
    df = re.sub(r"\'ll", " will", df)
    df = re.sub(r"\'ve", " have", df)
    df = re.sub(r"\'re", " are", df)
    df = re.sub(r"\'d", " would", df)
    df = re.sub(r"[-()#/@;:<>~{}+=?.|,]", "", df )

    return df

"""
Função para para fazer o pre-processamento do dataset 

@return -> database de movies-lines convertido
@return -> database de movie-convertation convertidas

"""
def preProcessing (  ) :

    dfML = open("movie-lines.txt", encoding="utf-8", errors="ignore").read().split("\n")
    dfMC = open("movie-conversations.txt", encoding="utf-8", errors="ignore").read().split("\n")

    i = 0

    # criando um dicionário das conversas
    idDfML = {}
    conversasId = []

    for linha in dfML :

        _linha = linha.split ( " +++$+++ " )

        if (len(_linha) == 5) :

            idDfML[_linha[0]] = _linha[4]

    # o último registro é excluido, pois ele possui apenas um caracter vazio
    for conversa in dfMC[:-1] :

        # pegando os ids das conversas
        _conversa = conversa.split( " +++$+++ " )[-1][1:-1].replace("'", "").replace(" ", "")

        # pegando os textos de id para textos
        conversasId.append(_conversa.split(","))

    # separação das perguntas e respostas
    perguntas, respostas = [], []

    for conversa in conversasId :

        for i in range(0, len ( conversa ) - 1 ) :

            perguntas.append ( idDfML [ conversa [ i ] ] )
            respostas.append ( idDfML [ conversa [ i + 1 ] ] )

    perguntasLimpas, respostasLimpas = [], []

    for pergunta in perguntas :

        perguntasLimpas.append ( limpaTexto ( pergunta ) )

    for resposta in respostas :

        respostasLimpas.append ( limpaTexto ( resposta ) )

    # dicionário para vermos quais são as palavras mais importantes para o treinamento do algorítmo
    palavrasContagem = {}

    for pergunta in perguntasLimpas :

        for palavra in pergunta.split (  ) :

            if ( palavra not in palavrasContagem ) :

                palavrasContagem [ palavra ] = 1

            else :

                palavrasContagem [ palavra ] += 1

    for resposta in respostasLimpas :

        for palavra in resposta.split (  ) :

            if ( palavra not in palavrasContagem ) :

                palavrasContagem [ palavra ] = 1

            else :

                palavrasContagem [ palavra ] += 1

    # criando o dicionário das perguntas
    numeroPala = 0

    perguntaPalavrasInt = {}

    for palavra, contagem in palavrasContagem.items (  ) :

        if ( contagem >= 20 ) :

            perguntaPalavrasInt [ palavra ] = numeroPala
            numeroPala += 1

    # criando o dicionário das respostas
    numeroPala = 0

    respostaPalavrasInt = {}

    for palavra, contagem in palavrasContagem.items():

        if ( contagem >= 20 ) :

            respostaPalavrasInt [ palavra ] = numeroPala
            numeroPala += 1

    # adição de tokens no dicionário
    tokens = ["<PAD>", "<EOS>", "<OUT>", "<SOS>"]

    for token in tokens :

        perguntaPalavrasInt [ token ] = len ( perguntaPalavrasInt ) + 1

    for token in tokens :

        respostaPalavrasInt [ token ] = len ( respostaPalavrasInt ) + 1

    # invertendo a chave com o dicionário
    respostaIntPalavras = {pI : p for p, pI in respostaPalavrasInt.items()}

    # adição do token final da string <EOS> para o final da resposta
    for i in range ( len ( respostasLimpas ) ) :

            respostasLimpas [ i ] += " <EOS>"

    # tradução de todas as perguntas e respostas para inteiros
    # substituição das palavras menos frequentes para <OUT>
    perguntasParaInt = []
    for pergunta in perguntasLimpas :

        ints = []

        for palavra in pergunta.split() :

            if ( palavra not in perguntaPalavrasInt ) :

                ints.append ( perguntaPalavrasInt [ "<OUT>" ] )

            else :

                ints.append ( perguntaPalavrasInt [ palavra ] )

        perguntasParaInt.append ( ints )

    respostasParaInt = []
    for resposta in respostasLimpas :

        ints = []

        for palavra in resposta.split():

            if (palavra not in respostaPalavrasInt):

                ints.append(respostaPalavrasInt[ "<OUT>" ])

            else :

                ints.append(respostaPalavrasInt[palavra])

        respostasParaInt.append ( ints )

    # ordenação das perguntas e respostas pelo tamanho das perguntas

    perguntasLimpasOrdenadas, respostasLimpasOrdenadas = [], []
    for tamanho in range ( 1, 26 ) :

        for i in enumerate ( perguntasParaInt ) :

            if ( len ( i [ 1 ] ) == tamanho ) :

                perguntasLimpasOrdenadas.append ( perguntasParaInt [ i [ 0 ] ] )
                respostasLimpasOrdenadas.append ( respostasParaInt [ i [ 0 ] ] )
    # precisa ainda arrumar o que o modelo vai retornar
    return dfML, dfMC

"""

Criação dos placeholders para as entradas e saídas

Modelo com dimensão [64,25]

    :return -> parâmetros da rede neural

"""
def entradaModelo() :

    entradas = tf.placeholder(tf.int32, [None, None], name = "entradas")
    saidas = tf.placeholder(tf.int32, [None, None], name = "saidas")
    learRate = tf.placeholder(tf.float32, name = "learning_rate")
    keepProb = tf.placeholder(tf.float32, name = "keep_prob") # dropout da rede

    return entradas, saidas, learRate, keepProb


"""

pré-processamento das saídas
colocando SOS no início das palavras

será criado também uma matriz da forma
[batchSize, 1] = [64, 1]

nessa coluna será adicionado o valor 
0 -> <SOS> (8825) -> id da palavra SOS
1 -> <SOS> (8825)

Ou seja, a cada 64 dados, será colocado esse caracter <SOS> no começo da palavra 

    :param
        saidas = resposta de saída
        palavraParaInt = dicionário que contém as palavras para inteiro
        batchSize = batch size da função que será definido como 64

    :return -> saidas pré-processadas

"""
def preProcessingSaidas(saidas, palavraParaInt, batchSize) :

    esquerda = tf.fill([batchSize, 1], palavraParaInt["<SOS>"])
    direita = tf.strided_slice(saidas, [0, 0], [batchSize, -1], strides = [1, 1]) # remoção dos <EOS>

    # concatenando as variáveis por coluna
    saidasPross = tf.concat([esquerda, direita], axis = 1)

    return saidasPross

"""

Criação da camada RNN do codificador

    :param
        rnnEntradas = recebe o retorno dos parâmetros da função entradaModelo
        rnnTamanho = número de tensores de entrada  ( recebimento dos texto )
        numCamadas = número de camadas
        keepProb = dropout da rede
        tamSeq = indica o tamanho da sequência

    :return -> retorna o enconder do estado

"""
def rnnCodificador(rnnEntradas, rnnTamanho, numCamadas, keepProb, tamSeq) :

    Lstm = tf.compat.v1.nn.rnn_cell.LSTMCell(rnnTamanho)

    # criação do drop out
    lstmDropOut = tf.compat.v1.nn.rnn_cell.DropoutWrapper(Lstm, input_keep_prob = keepProb)

    # criação de várias camadas
    encoderCelulas = tf.compat.v1.nn.rnn_cell.MultiRNNCell([lstmDropOut] * numCamadas)

    # deixar como observação que pode dar problema
    enconderEstado = tf.nn.bidirectional_dynamic_rnn ( cell_fw = encoderCelulas,
                                                       cell_bw = encoderCelulas,
                                                       sequence_lenght = tamSeq,
                                                       inputs = rnnEntradas,
                                                       dtype = tf.float32)

    return enconderEstado

"""

Decodificador da base de treinamento

    :param
        enconderEstado = retorno da função rnnCodificador
        decodCelula = decodificador da célula
        decodEMdEntrada = valores já decodificados
        tamSeq = tamanho da sequência das frases
        decodEscopo = escopo de varíaveis
        funSaida = função de saída -> estimators
        keepProb = Dropout da rede
        batchSize = batch size da rede neural 

"""
def decodBaseTrein(enconderEstado, decodCelula, decodEMdEntrada, tamSeq,
                   decodEscopo, funSaida, keepProb, batchSize) :

    estadosAtencao = tf.zeros([batchSize, 1, decodCelula.output_size])
    attKeys, attValues, attScoreFun, attConstrucFunc = tf.contrib.seq2seq.DynamicAttentionWapper(estadosAtencao,
                                                                                                 attention_option = "bahdanau",
                                                                                                 num_units = decodCelula.output_size)
    funcDecoTrain = tf.contrib.seq2seq.attention_decoder_fn_train(enconderEstado[0],
                                                                  attKeys,
                                                                  attValues,
                                                                  attScoreFun,
                                                                  attConstrucFunc,
                                                                  name = "attn_dec_train")

    decSaida, _, _ = tf.contrib.seq2seq.dynamic_rnn_decoder


def main (  ) :

    #dfML, dfMC = preProcessing (  )

    print(tf.__version__)

if __name__ == '__main__':
    main (  )