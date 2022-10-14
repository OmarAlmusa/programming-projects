function fib(n, memo = {}){
    if (n in memo) return memo[n];
    if (n <= 2) return 1;
    memo[n] = fib(n-1, memo) + fib(n-2, memo);
    return memo[n];
}

function gridtraveler(x, y, memo = {}){
    const key = x + ',' + y;
    if (key in memo) return memo[key];
    if (x == 1 && y == 1) return 1;
    if (x <= 0  || y <= 0) return 0;
    memo[key] = gridtraveler(x-1, y, memo) + gridtraveler(x, y-1, memo)
    return memo[key];

}

function cansum(target, array, memo = {}){
    if (target in memo) return memo[target];
    if (target === 0) return true;
    if (target <0) return false;

    for(let e of array){
        memo[target] = cansum(target - e, array, memo);
        if (memo[target] === true) return true;
    }
    
    return false
}

function howsum(target, array, memo = {}){
    if (target in memo) return memo[target];
    if (target === 0) return [];
    if (target < 0) return null;

    for(let e of array){
        memo[target] = howsum(target - e, array, memo);
        if (memo[target] !== null) return [...memo[target], e];
    }

    return null;

}

//console.log(howsum(7, [2, 3]));
//console.log(howsum(7, [5, 3, 4, 7]));
//console.log(howsum(7, [2, 4]));
//console.log(howsum(8, [2, 3, 5]));
//console.log(howsum(300, [7, 14]));

function bestsum(target, array, iter = {}){
    if (target in iter) return iter[target];
    if (target === 0) return 1;
    if (target < 0) return null;

    for(let e of array){
        if (bestsum(target, array, iter) !== null){
            iter[target] = iter[target] + 1;
        }
    }
    return Math.min(...Object.values(iter))
    //return null;
}


console.log(bestsum(7, [2, 3]));
console.log(bestsum(7, [5, 3, 4, 7]));
console.log(bestsum(7, [2, 4]));
console.log(bestsum(8, [2, 3, 5]));
console.log(bestsum(300, [7, 14]));