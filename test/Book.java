class Book extends Product{
    String name;
    String author;
    Price price;

    public void discount(Campaign campaign) {
        double discountPrice = price.getCurrentValue();
        discountPrice = discountPrice * campaign.getDiscountRate();
        return discountPrice;
    }
}