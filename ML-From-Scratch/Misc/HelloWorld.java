public class HelloWorld {
    
    void main() {
        String[] s = {"a", "b", "c"}; 
        sort(s);
    }

    public void sort(String[] x) {
        if(x[0] > x[1]) {
            System.out.println("yes");
        } else System.out.println("no");

    }
}
