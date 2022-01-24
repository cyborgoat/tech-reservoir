const printMyName = (name, age) => {
    console.log(name);
    console.log(age);
}

printMyName("Hello", 28);

class Human {
    gender = 'male';
    printGender = () => {
        console.log(this.gender);
    }
}

class Person extends Human {
    name = 'max';
    gender = 'female';

    printMyName = () => {
        console.log(this.name);
    }
}

const person2 = new Person();
person2.printMyName();
person2.printGender();