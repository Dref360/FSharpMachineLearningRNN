// Learn more about F# at http://fsharp.org
// See the 'F# Tutorial' project for more help.
open System
open System.IO

open Hype
open Hype.Neural
open Hype.NLP
open DiffSharp.AD.Float32
open DiffSharp.Util



//let addText text file = File.ReadAllText(file)
//let text = addText "" "C:\\test.txt"
//let text = "I sing of arms and the man, he who, exiled by fate, first came from the coast of Troy to Italy, and to Lavinian shores – hurled about endlessly by land and sea, by the will of the gods, by cruel Juno’s remorseless anger, long suffering also in war, until he founded a city and brought his gods to Latium: from that the Latin people came, the lords of Alba Longa, the walls of noble Rome. Muse, tell me the cause: how was she offended in her divinity, how was she grieved, the Queen of Heaven, to drive a man, noted for virtue, to endure such dangers, to face so many trials? Can there be such anger in the minds of the gods?"
let text = File.ReadAllText("C:\\test.txt")
let lang = Language(text)
lang.Tokens |> printfn "%A"
lang.Length |> printfn "%A"

let text' = lang.EncodeOneHot(text)

let data = Dataset(text'.[*, 0..(text'.Cols - 2)],
                    text'.[*, 1..(text'.Cols - 1)])
let dim = lang.Length // Vocabulary size

let train n path = 
    printfn "%s" path
    for i = 0 to 1000 do
        let par = {Params.Default with
                    //Batch = Minibatch 10
                    LearningRate = LearningRate.RMSProp(D 0.01f, D 0.9f)
                    Loss = CrossEntropyOnSoftmax
                    Epochs = i
                    Silent = true       // Suppress the regular printing of training progress
                    ReturnBest = false} 
        let loss, _ = Layer.Train(n, data, par)
        let sampleString = (lang.Sample(n.Run, "I", [|"."|], 30))
        printfn "Epoch: %*i | Loss: %O | Sample: %s" 3 i loss sampleString
        File.AppendAllText (path, (String.Format ("Epoch: {0} | Loss: {1} | Sample: {2}", i, loss, sampleString)))


[<EntryPoint>]
let main argv = 

    lang.Tokens |> printfn "%A"
    lang.Length |> printfn "%A"
    text'.Visualize() |> printfn "%s"

    
    let n = FeedForward()
    n.Add(Linear(dim, 20))
    n.Add(LSTM(20, 100))
    n.Add(LSTM(100, 100))
    n.Add(Linear(100, dim))
    n.Add(DM.mapCols softmax)


    train n "lstm2x.log"

    let n2 = FeedForward()
    n2.Add(Linear(dim, 20))
    n2.Add(LSTM(20, 100))
    n2.Add(GRU(100, 100))
    n2.Add(Linear(100, dim))
    n2.Add(DM.mapCols softmax)

    train n2 "lstmgru.log"

    0

