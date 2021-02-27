
use crate::State;
use tide::http::mime;

pub async fn index(mut _req: tide::Request<State>) -> tide::Result {
	// app.at("/").get(|_| async { Ok("visit /src/*") });
	// app.at("/src").serve_dir("src/")?;
	// app.at("/example").serve_file("examples/static_file.html")?;

	//app.at("/fib/:n").get(fibsum);
	//req.param("n")?.parse().unwrap_or(0);

	let res = tide::Response::builder(200)
		.body(include_str!("index.html"))
		//.header("Content-Type", "text/html; charset=UTF-8")
		.content_type(mime::HTML)
		.build();
	Ok(res)
}