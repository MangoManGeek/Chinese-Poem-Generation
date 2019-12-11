from generate import Generator
from plan import Planner

from eval_rhyme.evaluate import eval_train_data

if __name__ == '__main__':
    planner = Planner()
    generator = Generator()
    plan_keywords=[]

    with open('data/plan_data.txt') as f_in:
        while True:
            line=f_in.readline()
            for w in line.split():
                plan_keywords.append(w)
            if len(plan_keywords)>50:
                break
    print(plan_keywords)

    poems=[]

    for hints in plan_keywords:
        #hints = input("Type in hints >> ")
        keywords = planner.plan(hints)
        print("Keywords: " + ' '.join(keywords))
        poem = generator.generate(keywords)
        poems.append(poem)
        print("Poem generated:")
        for sentence in poem:
            print(sentence)

    scores, mean_score, std_score=eval_train_data(poems)

    for s in scores:
        print (s)






    