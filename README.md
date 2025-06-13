# Facial Expression Recognition Project


## Project Structure
პროექტი დაყოფილია src და notebook კომპონენტებად, 
src/data_utils.py - დატის ჩატვირთვა და პრეპროცესინგის ეტაპისთვის საჭირო მეთოდები. 
src/models.py - მოდელების არქიტექტურები კლასებად დაყოფილი.
src/training_utils.py - ტრეინინგისა და ოპტიმიზაციისთვის საჭირო მეთოდები
src/evaluation.py - მეტრიკებისა და ვიზუალიზაციების გამოსაჩენად საჭირო მეთოდები.

baseline mlp - 3 შრიანი მარტივი FCN პირველადი შედეგებისთვის
advanced mlp - გაუმჯობესებული FCN Batch normalization, dropout regularization და Xavier initialization-ით
advanced cnn - residual network-ია საწყისი conv layer-ით, residual block-ებით, global average pooling-ით, დროფაუთითა და FC-ით

## Notebooks
baseline_mlp.ipynb - თავდაპირველად გავუშვი k-fold-ზე გარკვეული default პარამეტრებით, მივიღე საშუალოდ 33% სიზუსტე ვალიდაციაზე, შემდგომ შემოვიღე სხვადასხვა ჰიპერპარამეტრების კონფიგურაციები, ოღონდ წარმოვადგინე გრიდის სახით და მათი ყველა შესაძლო კომბინაციიდან 10 რანდომ გავტესტე (ისე ბევრი გამოვიდოდა და 10 ჩავთვალე საკმარისი იყო ზოგადი სურათის სანახავად, თან baseline მოდელი ისედაც ვერ იფრენდა), შედეგად აირჩა საუკეთესო მოდელი და მისი ტრენინგის შედეგად მივიღე 35% სიზუსტე ვალიდაციაზე. confusion მატრიცის აგების შედეგად გამოჩნდა, რომ მოდელს ჰქონდა bias, კერძოდ happy-ს პასუხობდა ძალიან ხშირად.

improved_mlp.ipynb - არქიტექტურა გაიტესტა სხვადასხვა აქტივაციის ფუნქციაზე, დაძაბულ ბრძოლაში ჩემმა პირადმა ფავორიტმა leaky_relu-მ იმარჯვა, უკვე იგრძნობოდა ამ არქიტექტურის უპირატესობა baseline მოდელზე 40%-იანი სიზუსტით. შევადარე სხვადასხვა optimizer-ები, რაოდენ გასაკვირიც არ უნდა იყოს sgd საუკეთესო აღმოჩნდა. შევადარე ბაჩნორმის გამოყენება ჯობდა თუ არა და აღმოჩნდა რომ - კი. საბოლოო მოდელი ჩამოყალიბდა ამ კონფიგურაციით და ვალიდაციაზე მივიღე 40%, loss და validation accuracy ნორმალურად გამოიყურებოდა. მოდელის Pipeline-ის შენახვა მინდოდა მაგრამ რაღაცა მიერორებდა და ვერ გავასწორე.

simple_cnn.ipynb - data augmentation experiment - ში გაიტესტა 3 augmentation, გაიტესტა ასევე 2 არქიტექტურა ერთი უბრალოდ დაბალი დროფაუთით და მეორე ბაჩნორმით, ასევე გაიტესტა სხვადასხვა ფილტრების კონფიგურაციები (ზომები და რაოდენობები), შედეგად აირჩა მათ შორის საუკეთესო და ჩამოყალიბდა საბოლოო მოდელი, საწყისი პარამეტრები საკმარისი აღმოჩნდა, რომ მოდელს ოპტიმალურად ემუშავა და 10 ეპოქის შედეგად მივიღეთ 57% acc ვალიდაციაზე. ასევე confusion matrix-ზე დაკვირვებით შეიმჩნევა, რომ მაგალითდ Fear და Sad ერთმანეთში ხშირად ირეოდა. Loss და Accuracy-ს გრაფიკები Training და Validation-ზე კარგად გამოიყურება. საბოლოო ტრენინგის დროს 50 ეპოქაზე მივიღე 61% სიზუსტე ვალიდაციაზე.

advanced_cnn.ipynb - 3 augmentation გაიტესტა, ასევე optimizer-ები, რომლებიც გავიარეთ, შემოწმდა 3 სხვადასხვა დროფაუთი, საუკეთესო მოდელი შეირჩა ამ კონფიგურაციით, თუმცა ეს არქიტექტურა ბევრად რთული აღმოჩნდა ვიდრე დატა, გამოვიყენე ResNet არქიტექტურა (შედეგად 6 conv layer გამიჩნდა), ამის გამო ჰქონდა საკმაოდ დიდი overfitting. თავდაპირველად ვცადე dropout-ის გაზრდა, learning rate-ის შემცირება, ასევე შემოვიღე weight_decay ადამში და დავამატე learning rate scheduler (scheduler-ს ჩემი დასკვნებით არანაირი გავლენა არ ჰქონდა ამ მოდელზე), მიუხედავად იმისა, რომ overfitting შემცირდა, მოდელი ვერ ახერხებდა generalization-ს, რის გამოც learning rate გავუზარდე, dropout დავუგდე, ასევე adam-ის მაგივრად გამოვიყენე აdamw (მაინც decay-ს გამო, რა იცი რა ხდება) თუმცა ამჯერად უკვე overfit-ში გადიოდა. ვცადე batch_size-ის შემცირება ეგებ და დატა იყო უცნაურად დალაგებული, არ ჰქონია შედეგი, საბოლოოდ dropout გავზარდე 0.9-მდე, ანუ უკიდურესობა განვიხილე და მაინც overfit ჰქონდა რამაც საბოლოოდ დამარწმუნა, რომ არქიტექტურა იყო საკმაოდ კომპლექსური. ამ არქიტექტურას დაახლოებით იგივე შედეგი ჰქონდა ვალიდაციაზე, რაც simple_cnn-ს და რადგან overfit არ ჰქონდა simple-ს გადავწყვიტე მაინც simple_cnn-ის საუკეთესო მოდელად არჩევა (ბევრ ეპოქაზე არ გამიშვია ეგ, 10-ზე და პოტენციაში შეიძლება უკეთესიც ყოფილიყო ვიდრე advanced)


საბოლოო ჯამში simple_cnn გამოდგა საუკეთესო 61% სიზუსტით ტესტ-სეტზე
