use('StockData');

db.NewsArticles.find().forEach(function(doc) {
    var dateString = doc.Date;
    var newDate = new Date(dateString);
    db.NewsArticles.updateOne({_id: doc._id}, {$set: {Date: newDate}});
});
