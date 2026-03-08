import pandas as pd


df = pd.read_csv('train.csv')



def prepare_data(df):
    df['Name'] = df['Name'].fillna('Unknown Unknown')
    df['LastName'] = df['Name'].str.split().str[-1]

    df['Group'] = df['PassengerId'].str.split('_').str[0]
    df['Deck'] = df['Cabin'].str[0]
    df['Side'] = df['Cabin'].str[-1]

    df['GroupSize'] = df.groupby('Group')['Group'].transform('count')
    df['FamilySize'] = df.groupby('LastName')['LastName'].transform('count')

    numeric = ['Age', 'RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
    for i in numeric:
        df[i] = df[i].fillna(df[i].median())


    categorical = ['HomePlanet', 'CryoSleep', 'Cabin', 'Destination', 'VIP', 'Deck', 'Side']
    for i in categorical:
        if i in df.columns:
            df[i] = df[i].fillna(df[i].mode()[0])


    bool_cols = ['CryoSleep', 'VIP', 'Transported']
    for col in bool_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    df['Total_spent'] = df['RoomService'] + df['FoodCourt'] + df['ShoppingMall'] + df['Spa'] + df['VRDeck']


    df = df.drop(columns=['PassengerId', 'Name', 'Cabin', 'Group', 'LastName'])

    df = pd.get_dummies(df, columns=['HomePlanet', 'Destination', 'Deck', 'Side'])

    return df



train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
test_ids = df_test['PassengerId']


train = prepare_data(train)
test_processed = prepare_data(df_test)


X = train.drop(columns=['Transported'])
y = train['Transported']


from catboost import CatBoostClassifier
model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    eval_metric='Accuracy',
    verbose=100
)
model.fit(X, y)
final_preds = model.predict(test_processed)

submission = pd.DataFrame({
    "PassengerId": test_ids,
    "Transported": final_preds.astype(bool)
})

submission.to_csv('submission.csv', index=False)
print("Ready")
