{% extends 'layout.html' %}

{% block body %}

<?php
include('./Classes/DB.php');
$drug_name = 'Carbamazepine';
$reviews = DB::query('SELECT * FROM Reviews WHERE Drug=:drug Limit 20', array(':drug'=>$drug_name));
// var_dump($reviews);

foreach ($reviews as $review ) {

  // code...

  echo $review['Review'] . '<br>';
}



 ?>

 {% endblock %}
