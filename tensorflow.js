// import * as tf from '@tensorflow/tfjs';
const tf = require('@tensorflow/tfjs-node')
const fs = require('fs');
const path = require("path");

function getAllFiles(dir) {
    var files = [];
    fs.readdirSync(dir).forEach(file => {
        const absolute = path.join(dir, file);
        if (fs.statSync(absolute).isDirectory()) {
            files = files.concat(getAllFiles(absolute));
            }
        else{
            files.push(absolute);
        }
    });
    return files;
}

function formatData(groups){
    var dataArray = {
        inputs: [],
        outputs: [],
    };
    for (let i = 0; i < groups.length; i++){
        var filepath = groups[i].filepath;
        var label = groups[i].label;
        var allFiles = getAllFiles(filepath);
        var dataDictionary = {};
        allFiles.forEach(filename => {
            if (filename.split('.').pop() == "csv"){
                var fileContents = fs.readFileSync(filename, "ascii");
                var inputArray = fileContents.split("\n").map(x=> Math.abs(parseInt(x)));
                if (filename.includes("X__AXIS")){
                    var key = filename.replace("X__AXIS", "");
                    if (!dataDictionary.hasOwnProperty(key)){
                        dataDictionary[key] = {
                            x: inputArray,
                            y: null,
                            z: null,
                        }
                    } else {
                        dataDictionary[key].x = inputArray;
                    }
                } else if (filename.includes("Y__AXIS")){
                    var key = filename.replace("Y__AXIS", "");
                    if (!dataDictionary.hasOwnProperty(key)){
                        dataDictionary[key] = {
                            x: null,
                            y: inputArray,
                            z: null,
                        }
                    } else {
                        dataDictionary[key].y = inputArray;
                    }
                } else if (filename.includes("Z__AXIS")){
                    var key = filename.replace("Z__AXIS", "");
                    if (!dataDictionary.hasOwnProperty(key)){
                        dataDictionary[key] = {
                            x: null,
                            y: null,
                            z: inputArray,
                        }
                    } else {
                        dataDictionary[key].z = inputArray;
                    }
                }
            }
        });
        var keys = Object.keys(dataDictionary);
        for (let i = 0; i < keys.length; i++){
            if (dataDictionary[keys[i]].x != null && dataDictionary[keys[i]].y != null && dataDictionary[keys[i]].z != null){
                dataArray.inputs.push(
                    [dataDictionary[keys[i]].x, 
                    dataDictionary[keys[i]].y,
                    dataDictionary[keys[i]].z
                    ]
                );
                dataArray.outputs.push(
                    [label]
                );
            }
        }
    }
    return dataArray;
}



async function main(){
    var dataStreams = [
        {filepath: "C:\\Users\\david\\Desktop\\randomrest", label: 1},
        {filepath: "C:\\Users\\david\\Desktop\\Data\\Nonshots", label: 0}
    ];
    
    var data = formatData(dataStreams);
    
    var sampleSize = data.inputs.length;
    var inputDimension = data.inputs[0].length;
    console.log("inputDimension:", inputDimension);
    var inputLength = data.inputs[0][0].length;
    console.log("inputLength: ", inputLength);
    
    // Define a model for linear regression.
    const model = tf.sequential();
    model.add(tf.layers.dense({units: inputLength, inputShape: [inputDimension, inputLength], useBias: true, activation: 'relu6'}));
    model.add(tf.layers.dense({units: inputLength, activation: 'relu6'}));
    model.add(tf.layers.dense({units: 1, activation: 'sigmoid'}));
    
    var xs = tf.tensor(data.inputs, [sampleSize, inputDimension, inputLength]);
    console.log("xs shape: ", xs.shape);
    var ys = tf.tensor(data.outputs, [sampleSize, 1]);
    console.log("ys shape: ", ys.shape);
    
    // Prepare the model for training: Specify the loss and the optimizer.
    model.compile({loss: 'meanSquaredError', optimizer: 'sgd'});

    model.weights.forEach(w => {
        console.log(w.name, w.shape);
       });
    
    for (let i = 1; i < 5 ; ++i) {
        const h = await model.fit(xs, ys, {
            batchSize: 4,
            epochs: 3
        });
        console.log("Loss after Epoch " + i + " : " + h.history.loss[0]);
     }
    
    
    // // Train the model using the data.
    // model.fit(xs, ys).then(() => {
    //   // Use the model to do inference on a data point the model hasn't seen before:
    //   model.predict(tf.tensor2d([5], [1, 1])).print();
    // });
}

main();